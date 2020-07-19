#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "gp.h"
#include "bvh.h"

typedef struct gp_scene {
  gp_bvh       bvh;
  gp_material* materials;
  uint32_t     material_count;
} gp_scene;

void gp_fail(const char* msg)
{
  printf("Gatling encountered a fatal error: %s\n", msg);
  exit(-1);
}

void gp_load_scene(
  gp_scene* scene,
  const char* file_path)
{
  struct aiPropertyStore* props = aiCreatePropertyStore();
  aiSetImportPropertyInteger(props, AI_CONFIG_PP_FD_REMOVE, 1);

  const struct aiScene* ai_scene = aiImportFileExWithProperties(
    file_path,
    aiProcess_Triangulate |
      aiProcess_FindInvalidData |
      aiProcess_GenSmoothNormals |
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

  uint32_t vertex_count = 0;
  uint32_t face_count = 0;

  for (uint32_t m = 0; m < ai_scene->mNumMeshes; ++m)
  {
    const struct aiMesh* ai_mesh = ai_scene->mMeshes[m];
    vertex_count += ai_mesh->mNumVertices;
    face_count += ai_mesh->mNumFaces;
  }

  gp_vertex* vertices =
    (gp_vertex*) malloc(vertex_count * sizeof(gp_vertex));
  gp_face* faces =
    (gp_face*) malloc(face_count * sizeof(gp_face));

  uint32_t vertex_index = 0;
  uint32_t face_index = 0;

  for (uint32_t m = 0; m < ai_scene->mNumMeshes; ++m)
  {
    const struct aiMesh* ai_mesh = ai_scene->mMeshes[m];

    for (uint32_t f = 0; f < ai_mesh->mNumFaces; ++f)
    {
      const struct aiFace* ai_face = &ai_mesh->mFaces[f];
      assert(ai_face->mNumIndices == 3);

      struct gp_face* face = &faces[face_index];

      face->v_i[0] = vertex_index + ai_face->mIndices[0];
      face->v_i[1] = vertex_index + ai_face->mIndices[1];
      face->v_i[2] = vertex_index + ai_face->mIndices[2];
      face->mat_index = ai_mesh->mMaterialIndex;

      face_index++;
    }

    for (uint32_t v = 0; v < ai_mesh->mNumVertices; ++v)
    {
      const struct aiVector3D* ai_position = &ai_mesh->mVertices[v];
      const struct aiVector3D* ai_normal = &ai_mesh->mNormals[v];
      const struct aiVector3D* ai_tex_coords = &ai_mesh->mTextureCoords[0][v];

      struct gp_vertex* vertex = &vertices[vertex_index];

      vertex->pos[0] = ai_position->x;
      vertex->pos[1] = ai_position->y;
      vertex->pos[2] = ai_position->z;
      vertex->norm[0] = ai_normal->x;
      vertex->norm[1] = ai_normal->y;
      vertex->norm[2] = ai_normal->z;
      vertex->uv[0] = 0.0f;
      vertex->uv[1] = 0.0f;

      vertex_index++;
    }
  }

  const gp_bvh_build_params bvh_params = {
    .face_batch_size          = 1,
    .face_count               = face_count,
    .face_intersection_cost   = 1.2f,
    .faces                    = faces,
    .leaf_max_face_count      = 4,
    .object_binning_enabled   = true,
    .object_binning_threshold = 1024,
    .object_bin_count         = 16,
    .vertex_count             = vertex_count,
    .vertices                 = vertices
  };

  gp_bvh_build(
    &bvh_params,
    &scene->bvh
  );

  scene->material_count = ai_scene->mNumMaterials;
  scene->materials =
    (gp_material*) malloc(scene->material_count * sizeof(gp_material));

  for (uint32_t m = 0; m < ai_scene->mNumMaterials; ++m)
  {
    const struct aiMaterial* ai_mat = ai_scene->mMaterials[m];
    gp_material* material = &scene->materials[m];

    struct aiColor4D ai_color = { 1.0f, 0.0f, 1.0f, 0.0f };
    aiGetMaterialColor(ai_mat, AI_MATKEY_COLOR_DIFFUSE, &ai_color);
    material->r = ai_color.r;
    material->g = ai_color.g;
    material->b = ai_color.b;
    material->a = ai_color.a;
  }

  free(vertices);
  free(faces);

  aiReleaseImport(ai_scene);
}

void gp_write_file(
  const uint8_t* data,
  uint64_t byte_count,
  const char* file_path)
{
  FILE *file = fopen(file_path, "wb");
  if (file == NULL) {
    gp_fail("Unable to open file for writing.");
  }

  const uint64_t written_size = fwrite(data, 1, byte_count, file);
  if (written_size != byte_count) {
    gp_fail("Unable to write file.");
  }

  const int close_result = fclose(file);
  if (close_result != 0) {
    printf("Unable to close file '%s'.", file_path);
  }
}

void gp_free_scene(gp_scene* scene)
{
  gp_free_bvh(&scene->bvh);
  free(scene->materials);
}

uint32_t round_to_buffer_offset_alignment(uint32_t byte_offset)
{
  /* For now, since we upload one buffer and describe offsets into
   it, we must adhere to the device buffer offset alignment rules
   (e.g. Vulkan minStorageBufferOffsetAlignment device limit). At
   a later stage, we will mmap parts of the file and copy them into
   the GPU buffer with the required device alignment offsets dynamically.
   64 bytes will cover most discrete GPUs, but not iGPUs in smartphones. */
  const uint32_t required_offset_alignment = 64u;

  return (byte_offset + required_offset_alignment - 1) /
           required_offset_alignment * required_offset_alignment;
}

void gp_write_scene(
  const gp_scene* scene,
  const char* file_path)
{
  const gp_bvh* bvh = &scene->bvh;

  const uint32_t header_size_in_bytes = 56u;

  const uint32_t node_offset =
    round_to_buffer_offset_alignment(header_size_in_bytes);
  const uint32_t face_offset =
    round_to_buffer_offset_alignment(node_offset + bvh->node_count * sizeof(gp_bvh_node));
  const uint32_t vertex_offset =
    round_to_buffer_offset_alignment(face_offset + bvh->face_count * sizeof(gp_face));
  const uint32_t material_offset =
    round_to_buffer_offset_alignment(vertex_offset + bvh->vertex_count * sizeof(gp_vertex));

  const uint32_t total_size =
    material_offset + scene->material_count * sizeof(gp_material);

  uint8_t* buffer = malloc(total_size);

  memcpy(buffer +  0, &node_offset,             4);
  memcpy(buffer +  4, &bvh->node_count,         4);
  memcpy(buffer +  8, &face_offset,             4);
  memcpy(buffer + 12, &bvh->face_count,         4);
  memcpy(buffer + 16, &vertex_offset,           4);
  memcpy(buffer + 20, &bvh->vertex_count,       4);
  memcpy(buffer + 24, &material_offset,         4);
  memcpy(buffer + 28, &scene->material_count,   4);
  memcpy(buffer + 32, &bvh->aabb, sizeof(gp_aabb));

  memcpy(
    buffer + node_offset,
    bvh->nodes,
    bvh->node_count * sizeof(gp_bvh_node)
  );

  memcpy(
    buffer + face_offset,
    bvh->faces,
    bvh->face_count * sizeof(gp_face)
  );

  memcpy(
    buffer + vertex_offset,
    bvh->vertices,
    bvh->vertex_count * sizeof(gp_vertex)
  );

  memcpy(
    buffer + material_offset,
    scene->materials,
    scene->material_count * sizeof(gp_material)
  );

  gp_write_file(
    buffer,
    total_size,
    file_path
  );

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
