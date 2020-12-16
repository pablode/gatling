#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "gp.h"
#include "bvh.h"
#include "bvh_collapse.h"
#include "bvh_compress.h"

typedef struct gp_camera {
  gp_vec3 origin;
  gp_vec3 look_at;
  gp_vec3 up;
  float   hfov;
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

static void gp_assimp_add_node_mesh(
  const struct aiScene* ai_scene, const struct aiNode* ai_node,
  const struct aiMatrix4x4* ai_parent_transform,
  uint32_t* face_index, gp_face* faces,
  uint32_t* vertex_index, gp_vertex* vertices)
{
  struct aiMatrix4x4 ai_trans = *ai_parent_transform;
  aiMultiplyMatrix4(&ai_trans, &ai_node->mTransformation);

  struct aiMatrix3x3 ai_norm_trans;
  aiMatrix3FromMatrix4(&ai_norm_trans, &ai_trans);
  aiMatrix3Inverse(&ai_norm_trans);
  aiTransposeMatrix3(&ai_norm_trans);

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
      struct aiVector3D* ai_position = &ai_mesh->mVertices[v];
      struct aiVector3D* ai_normal = &ai_mesh->mNormals[v];

      aiTransformVecByMatrix4(ai_position, &ai_trans);
      aiTransformVecByMatrix3(ai_normal, &ai_norm_trans);

      struct gp_vertex* vertex = &vertices[*vertex_index];
      vertex->pos[0] = ai_position->x;
      vertex->pos[1] = ai_position->y;
      vertex->pos[2] = ai_position->z;
      vertex->norm[0] = ai_normal->x;
      vertex->norm[1] = ai_normal->y;
      vertex->norm[2] = ai_normal->z;
      vertex->uv[0] = 0.0f;
      vertex->uv[1] = 0.0f;

      gp_vec3_normalize(vertex->norm, vertex->norm);

      (*vertex_index)++;
    }
  }

  for (uint32_t i = 0; i < ai_node->mNumChildren; ++i)
  {
    gp_assimp_add_node_mesh(
      ai_scene, ai_node->mChildren[i], &ai_trans,
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

  struct aiMatrix4x4 ai_identity_matrix;
  aiIdentityMatrix4(&ai_identity_matrix);

  vertex_count = 0;
  face_count = 0;

  gp_assimp_add_node_mesh(
    ai_scene, ai_scene->mRootNode, &ai_identity_matrix,
    &face_count, faces, &vertex_count, vertices
  );

  scene->vertex_count = vertex_count;
  scene->vertices = realloc(vertices, vertex_count * sizeof(gp_vertex));

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

  if (ai_scene->mNumCameras == 0)
  {
    printf("Warning: no camera found\n");
  }
  else
  {
    struct aiCamera* ai_camera = ai_scene->mCameras[0];

    struct aiNode* ai_cam_node = gp_assimp_find_node(ai_scene->mRootNode, ai_camera->mName.data);
    struct aiMatrix4x4 ai_cam_trans = ai_cam_node->mTransformation;

    struct aiVector3D ai_origin = { 0.0f, 0.0f, 0.0f };
    aiTransformVecByMatrix4(&ai_origin, &ai_cam_trans);

    scene->camera.origin[0] = ai_origin.x;
    scene->camera.origin[1] = ai_origin.y;
    scene->camera.origin[2] = ai_origin.z;

    /* Remove position to transform directions. */
    ai_cam_trans.a4 = 0.0f;
    ai_cam_trans.b4 = 0.0f;
    ai_cam_trans.c4 = 0.0f;

    struct aiVector3D ai_look_at = ai_camera->mLookAt;
    struct aiVector3D ai_up = ai_camera->mUp;
    aiTransformVecByMatrix4(&ai_look_at, &ai_cam_trans);
    aiTransformVecByMatrix4(&ai_up, &ai_cam_trans);

    scene->camera.look_at[0] = ai_look_at.x;
    scene->camera.look_at[1] = ai_look_at.y;
    scene->camera.look_at[2] = ai_look_at.z;

    scene->camera.up[0] = ai_up.x;
    scene->camera.up[1] = ai_up.y;
    scene->camera.up[2] = ai_up.z;

    scene->camera.hfov = ai_camera->mHorizontalFOV;
  }

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
