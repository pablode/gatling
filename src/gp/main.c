#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <gml.h>

#include "gp.h"
#include "bvh.h"
#include "bvh_collapse.h"
#include "bvh_compress.h"

typedef struct gp_camera {
  gml_vec3 origin;
  gml_vec3 forward;
  gml_vec3 up;
  float    hfov;
} gp_camera;

typedef struct gp_scene {
  gp_bvhcc     bvhcc;
  gp_camera    camera;
  uint32_t     face_count;
  gp_face*     faces;
  gp_material* materials;
  uint32_t     material_count;
  uint32_t     vertex_count;
  gp_vertex*   vertices;
} gp_scene;

static void gp_fail(const char* msg)
{
  printf("Gatling encountered a fatal error: %s\n", msg);
  exit(-1);
}

static void gp_convert_assimp_mat4(const struct aiMatrix4x4* ai_mat, gml_mat4 mat)
{
  mat[0][0] = ai_mat->a1;
  mat[0][1] = ai_mat->a2;
  mat[0][2] = ai_mat->a3;
  mat[0][3] = ai_mat->a4;
  mat[1][0] = ai_mat->b1;
  mat[1][1] = ai_mat->b2;
  mat[1][2] = ai_mat->b3;
  mat[1][3] = ai_mat->b4;
  mat[2][0] = ai_mat->c1;
  mat[2][1] = ai_mat->c2;
  mat[2][2] = ai_mat->c3;
  mat[2][3] = ai_mat->c4;
  mat[3][0] = ai_mat->d1;
  mat[3][1] = ai_mat->d2;
  mat[3][2] = ai_mat->d3;
  mat[3][3] = ai_mat->d4;
}

static void gp_convert_assimp_vec3(const struct aiVector3D* ai_vec, gml_vec3 vec)
{
  vec[0] = ai_vec->x;
  vec[1] = ai_vec->y;
  vec[2] = ai_vec->z;
}

static void gp_assimp_add_node_mesh(
  const struct aiScene* ai_scene,
  const struct aiNode* ai_node,
  const gml_mat4 parent_trans,
  uint32_t* face_index, gp_face* faces,
  uint32_t* vertex_index, gp_vertex* vertices)
{
  gml_mat4 trans;
  gp_convert_assimp_mat4(&ai_node->mTransformation, trans);
  gml_mat4_mul(parent_trans, trans, trans);

  gml_mat3 norm_trans;
  gml_mat3_from_mat4(trans, norm_trans);
  gml_mat3_invert(norm_trans, norm_trans);
  gml_mat3_transpose(norm_trans, norm_trans);

  for (uint32_t m = 0; m < ai_node->mNumMeshes; ++m)
  {
    const struct aiMesh* ai_mesh = ai_scene->mMeshes[ai_node->mMeshes[m]];

    for (uint32_t f = 0; f < ai_mesh->mNumFaces; ++f)
    {
      const struct aiFace* ai_face = &ai_mesh->mFaces[f];
      assert(ai_face->mNumIndices == 3);

      struct gp_face* face = &faces[*face_index];
      face->v_i[0] = (*vertex_index) + ai_face->mIndices[0];
      face->v_i[1] = (*vertex_index) + ai_face->mIndices[1];
      face->v_i[2] = (*vertex_index) + ai_face->mIndices[2];
      face->mat_index = ai_mesh->mMaterialIndex;

      (*face_index)++;
    }

    for (uint32_t v = 0; v < ai_mesh->mNumVertices; ++v)
    {
      gml_vec3 pos;
      gml_vec3 normal;
      gp_convert_assimp_vec3(&ai_mesh->mVertices[v], pos);
      gp_convert_assimp_vec3(&ai_mesh->mNormals[v], normal);

      gml_vec4 pos4 = { pos[0], pos[1], pos[2], 1.0f };
      gml_mat4_mul_vec4(trans, pos4, pos4);

      gml_mat3_mul_vec3(norm_trans, normal, normal);
      gml_vec3_normalize(normal, normal);

      struct gp_vertex* vertex = &vertices[*vertex_index];
      vertex->pos[0] = pos4[0];
      vertex->pos[1] = pos4[1];
      vertex->pos[2] = pos4[2];
      vertex->norm[0] = normal[0];
      vertex->norm[1] = normal[1];
      vertex->norm[2] = normal[2];
      vertex->uv[0] = 0.0f;
      vertex->uv[1] = 0.0f;

      (*vertex_index)++;
    }
  }

  for (uint32_t i = 0; i < ai_node->mNumChildren; ++i)
  {
    gp_assimp_add_node_mesh(
      ai_scene, ai_node->mChildren[i], trans,
      face_index, faces, vertex_index, vertices
    );
  }
}

struct aiNode* gp_assimp_find_node(const struct aiNode* ai_parent, const char* name)
{
  for (uint32_t i = 0; i < ai_parent->mNumChildren; i++)
  {
    struct aiNode* ai_child = ai_parent->mChildren[i];

    if (!strcmp(ai_child->mName.data, name))
    {
      return ai_child;
    }

    struct aiNode* ai_target_node = gp_assimp_find_node(ai_child, name);

    if (ai_target_node)
    {
      return ai_target_node;
    }
  }

  return NULL;
}

static void gp_load_scene(gp_scene* scene, const char* file_path)
{
  /* Load scene using Assimp. */
  struct aiPropertyStore* props = aiCreatePropertyStore();
  aiSetImportPropertyInteger(props, AI_CONFIG_PP_FD_REMOVE, 1);

  const struct aiScene* ai_scene = aiImportFileExWithProperties(
    file_path,
    aiProcess_Triangulate |
      aiProcess_GenNormals |
      aiProcess_FindInvalidData |
      aiProcess_ImproveCacheLocality |
      aiProcess_JoinIdenticalVertices |
      aiProcess_TransformUVCoords |
      aiProcess_RemoveRedundantMaterials |
      aiProcess_FindDegenerates,
    NULL,
    props
  );

  aiReleasePropertyStore(props);

  if(!ai_scene)
  {
    const char* error_msg = aiGetErrorString();
    gp_fail(error_msg);
  }

  if ((ai_scene->mFlags & AI_SCENE_FLAGS_VALIDATION_WARNING) == AI_SCENE_FLAGS_VALIDATION_WARNING)
  {
    printf("Warning: Assimp validation warning\n");
  }
  if ((ai_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) == AI_SCENE_FLAGS_INCOMPLETE)
  {
    printf("Warning: Assimp scene import incomplete\n");
  }

  /* Get scene camera properties. */
  if (ai_scene->mNumCameras == 0)
  {
    printf("Error: no camera found\n");
    exit(EXIT_FAILURE);
  }
  else
  {
    gml_mat4 cam_trans;
    gml_mat4_identity(cam_trans);

    struct aiCamera* ai_camera = ai_scene->mCameras[0];
    struct aiNode* ai_cam_node = gp_assimp_find_node(ai_scene->mRootNode, ai_camera->mName.data);

    do
    {
      gml_mat4 node_trans;
      gp_convert_assimp_mat4(&ai_cam_node->mTransformation, node_trans);
      gml_mat4_mul(node_trans, cam_trans, cam_trans);
      ai_cam_node = ai_cam_node->mParent;
    }
    while (ai_cam_node);

    gml_vec4 origin = { 0.0f, 0.0f, 0.0f, 1.0f };
    gml_mat4_mul_vec4(cam_trans, origin, origin);
    scene->camera.origin[0] = origin[0];
    scene->camera.origin[1] = origin[1];
    scene->camera.origin[2] = origin[2];

    gml_mat3 dir_trans;
    gml_mat3_from_mat4(cam_trans, dir_trans);

    gp_convert_assimp_vec3(&ai_camera->mLookAt, scene->camera.forward);
    gp_convert_assimp_vec3(&ai_camera->mUp, scene->camera.up);
    gml_mat3_mul_vec3(dir_trans, scene->camera.forward, scene->camera.forward);
    gml_mat3_mul_vec3(dir_trans, scene->camera.up, scene->camera.up);
    gml_vec3_normalize(scene->camera.forward, scene->camera.forward);
    gml_vec3_normalize(scene->camera.up, scene->camera.up);

    scene->camera.hfov = ai_camera->mHorizontalFOV;
  }

  /* Calculate the inverse view matrix to transform
   * the whole scene graph into camera space. */
  gml_vec3 right;
  gml_vec3_cross(scene->camera.up, scene->camera.forward, right);

  gml_mat4 root_trans;
  root_trans[0][0] = right[0];
  root_trans[0][1] = right[1];
  root_trans[0][2] = right[2];
  root_trans[0][3] = -gml_vec3_dot(right, scene->camera.origin);
  root_trans[1][0] = scene->camera.up[0];
  root_trans[1][1] = scene->camera.up[1];
  root_trans[1][2] = scene->camera.up[2];
  root_trans[1][3] = -gml_vec3_dot(scene->camera.up, scene->camera.origin);
  root_trans[2][0] = scene->camera.forward[0];
  root_trans[2][1] = scene->camera.forward[1];
  root_trans[2][2] = scene->camera.forward[2];
  root_trans[2][3] = -gml_vec3_dot(scene->camera.forward, scene->camera.origin);
  root_trans[3][0] = 0.0f;
  root_trans[3][1] = 0.0f;
  root_trans[3][2] = 0.0f;
  root_trans[3][3] = 1.0f;

  scene->camera.up[0] = 0.0f;
  scene->camera.up[1] = 1.0f;
  scene->camera.up[2] = 0.0f;
  scene->camera.forward[0] = 0.0f;
  scene->camera.forward[1] = 0.0f;
  scene->camera.forward[2] = 1.0f;
  scene->camera.origin[0] = 0.0f;
  scene->camera.origin[1] = 0.0f;
  scene->camera.origin[2] = 0.0f;

  /* Calculate and allocate geometry memory. */
  uint32_t vertex_count = 0;
  uint32_t face_count = 0;

  for (uint32_t m = 0; m < ai_scene->mNumMeshes; ++m)
  {
    const struct aiMesh* ai_mesh = ai_scene->mMeshes[m];
    vertex_count += ai_mesh->mNumVertices;
    face_count += ai_mesh->mNumFaces;
  }

  gp_vertex* vertices = (gp_vertex*) malloc(vertex_count * sizeof(gp_vertex));
  gp_face* faces = (gp_face*) malloc(face_count * sizeof(gp_face));

  vertex_count = 0;
  face_count = 0;

  /* Transform and add scene graph geometry. */
  gp_assimp_add_node_mesh(
    ai_scene, ai_scene->mRootNode, root_trans,
    &face_count, faces, &vertex_count, vertices
  );

  scene->vertex_count = vertex_count;
  scene->vertices = realloc(vertices, vertex_count * sizeof(gp_vertex));

  /* Build BVH. */
  gp_bvh bvh;
  const gp_bvh_build_params bvh_params = {
    .face_batch_size          = 1,
    .face_count               = face_count,
    .face_intersection_cost   = 1.2f,
    .faces                    = faces,
    .leaf_max_face_count      = 1,
    .object_binning_mode      = GP_BVH_BINNING_MODE_FIXED,
    .object_binning_threshold = 1024,
    .object_bin_count         = 16,
    .spatial_bin_count        = 32,
    .spatial_reserve_factor   = 1.25f,
    .spatial_split_alpha      = 10e-5f,
    .vertex_count             = scene->vertex_count,
    .vertices                 = scene->vertices
  };

  gp_bvh_build(
    &bvh_params,
    &bvh
  );

  free(faces);

  gp_bvhc bvhc;
  gp_bvh_collapse_params cparams  = {
    .bvh                    = &bvh,
    .max_leaf_size          = 3,
    .node_traversal_cost    = 1.0f,
    .face_intersection_cost = 0.3f
  };

  gp_bvh_collapse(&cparams, &bvhc);
  gp_free_bvh(&bvh);

  scene->face_count = bvhc.face_count;
  scene->faces = malloc(bvhc.face_count * sizeof(gp_face));
  memcpy(scene->faces, bvhc.faces, bvhc.face_count * sizeof(gp_face));

  gp_bvh_compress(&bvhc, &scene->bvhcc);
  gp_free_bvhc(&bvhc);

  /* Read materials. */
  scene->material_count = ai_scene->mNumMaterials;
  scene->materials =
    (gp_material*) malloc(scene->material_count * sizeof(gp_material));

  for (uint32_t m = 0; m < ai_scene->mNumMaterials; ++m)
  {
    const struct aiMaterial* ai_mat = ai_scene->mMaterials[m];
    gp_material* material = &scene->materials[m];

    struct aiColor4D ai_albedo = { 1.0f, 0.0f, 1.0f, 0.0f };
    struct aiColor4D ai_emission = { 0.0f, 0.0f, 0.0f, 0.0f };
    aiGetMaterialColor(ai_mat, AI_MATKEY_COLOR_DIFFUSE, &ai_albedo);
    aiGetMaterialColor(ai_mat, AI_MATKEY_COLOR_EMISSIVE, &ai_emission);
    material->albedo_r = ai_albedo.r;
    material->albedo_g = ai_albedo.g;
    material->albedo_b = ai_albedo.b;
    material->padding1 = 0.0f;
    material->emission_r = ai_emission.r;
    material->emission_g = ai_emission.g;
    material->emission_b = ai_emission.b;
    material->padding2 = 0.0f;
  }

  /* Cleanup. */
  aiReleaseImport(ai_scene);
}

static void gp_write_file(
  const uint8_t* data,
  uint64_t size,
  const char* file_path)
{
  FILE *file = fopen(file_path, "wb");
  if (file == NULL) {
    gp_fail("Unable to open file for writing.");
  }

  const uint64_t written_size = fwrite(data, 1, size, file);
  if (written_size != size) {
    gp_fail("Unable to write file.");
  }

  const int close_result = fclose(file);
  if (close_result != 0) {
    printf("Unable to close file '%s'.", file_path);
  }
}

static void gp_free_scene(gp_scene* scene)
{
  gp_free_bvhcc(&scene->bvhcc);
  free(scene->materials);
  free(scene->vertices);
  free(scene->faces);
}

static void gp_write_scene(
  const gp_scene* scene,
  const char* file_path)
{
  const gp_bvhcc* bvhcc = &scene->bvhcc;

  const uint64_t header_size = 128;
  const uint64_t node_buf_offset = header_size;
  const uint64_t node_buf_size = bvhcc->node_count * sizeof(gp_bvhcc_node);
  const uint64_t face_buf_offset = node_buf_offset + node_buf_size;
  const uint64_t face_buf_size = scene->face_count * sizeof(gp_face);
  const uint64_t vertex_buf_offset = face_buf_offset + face_buf_size;
  const uint64_t vertex_buf_size = scene->vertex_count * sizeof(gp_vertex);
  const uint64_t material_buf_offset = vertex_buf_offset + vertex_buf_size;
  const uint64_t material_buf_size = scene->material_count * sizeof(gp_material);

  const uint64_t file_size = material_buf_offset + material_buf_size;

  uint8_t* buffer = malloc(file_size);

  memcpy(&buffer[ 0], &node_buf_offset,     8);
  memcpy(&buffer[ 8], &node_buf_size,       8);
  memcpy(&buffer[16], &face_buf_offset,     8);
  memcpy(&buffer[24], &face_buf_size,       8);
  memcpy(&buffer[32], &vertex_buf_offset,   8);
  memcpy(&buffer[40], &vertex_buf_size,     8);
  memcpy(&buffer[48], &material_buf_offset, 8);
  memcpy(&buffer[56], &material_buf_size,   8);
  memcpy(&buffer[64], &bvhcc->aabb,         sizeof(gp_aabb));
  memcpy(&buffer[88], &scene->camera,       sizeof(gp_camera));

  memcpy(&buffer[node_buf_offset], bvhcc->nodes, node_buf_size);

  memcpy(&buffer[face_buf_offset], scene->faces, face_buf_size);

  for (uint32_t i = 0; i < scene->vertex_count; ++i)
  {
    uint8_t* ptr = &buffer[vertex_buf_offset + i * 32];
    memcpy(&ptr[ 0], &scene->vertices[i].pos[0],  4);
    memcpy(&ptr[ 4], &scene->vertices[i].pos[1],  4);
    memcpy(&ptr[ 8], &scene->vertices[i].pos[2],  4);
    memcpy(&ptr[12], &scene->vertices[i].uv[0],   4);
    memcpy(&ptr[16], &scene->vertices[i].norm[0], 4);
    memcpy(&ptr[20], &scene->vertices[i].norm[1], 4);
    memcpy(&ptr[24], &scene->vertices[i].norm[2], 4);
    memcpy(&ptr[28], &scene->vertices[i].uv[1],   4);
  }

  memcpy(&buffer[material_buf_offset], scene->materials, material_buf_size);

  gp_write_file(buffer, file_size, file_path);

  free(buffer);
}

int main(int argc, const char* argv[])
{
  if (argc != 3)
  {
    printf("Usage: gp <input_file> <output.gsd>\n");
    return EXIT_FAILURE;
  }

  const char* file_path_in = argv[1];
  const char* file_path_out = argv[2];

  gp_scene scene;
  gp_load_scene(
    &scene,
    file_path_in
  );

  gp_write_scene(
    &scene,
    file_path_out
  );

  gp_free_scene(&scene);

  return EXIT_SUCCESS;
}
