#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "gp.h"

typedef struct gp_scene {
  gp_vertex*   vertices;
  uint32_t     vertices_count;
  gp_triangle* triangles;
  uint32_t     triangles_count;
} gp_scene;

GpResult gp_load_scene(
  gp_scene* p_scene,
  const char* p_file_path)
{
  const struct aiScene* ai_scene = aiImportFile(
    p_file_path,
    aiProcess_Triangulate |
      aiProcess_FindInvalidData |
      aiProcess_GenSmoothNormals |
      aiProcess_ImproveCacheLocality |
      aiProcess_JoinIdenticalVertices |
      aiProcess_TransformUVCoords |
      aiProcess_RemoveRedundantMaterials
  );

  if(!ai_scene)
  {
    const char* error_msg = aiGetErrorString();
    printf("Unable to import scene: %s\n", error_msg);
    return GP_UNABLE_TO_IMPORT_SCENE;
  }

  p_scene->vertices_count = 0;
  p_scene->triangles_count = 0;

  for (uint32_t m = 0; m < ai_scene->mNumMeshes; ++m)
  {
    const struct aiMesh* ai_mesh = ai_scene->mMeshes[m];
    p_scene->vertices_count += ai_mesh->mNumVertices;
    p_scene->triangles_count += ai_mesh->mNumFaces;
  }

  p_scene->vertices =
    (gp_vertex*) malloc(p_scene->vertices_count * sizeof(gp_vertex));
  p_scene->triangles =
    (gp_triangle*) malloc(p_scene->triangles_count * sizeof(gp_triangle));

  uint32_t vertex_index = 0;
  uint32_t triangle_index = 0;

  for (uint32_t m = 0; m < ai_scene->mNumMeshes; ++m)
  {
    const struct aiMesh* ai_mesh = ai_scene->mMeshes[m];

    for (uint32_t v = 0; v < ai_mesh->mNumVertices; ++v)
    {
      const struct aiVector3D* ai_position = &ai_mesh->mVertices[v];
      const struct aiVector3D* ai_normal = &ai_mesh->mNormals[v];
      const struct aiVector3D* ai_tex_coords = &ai_mesh->mTextureCoords[0][v];

      struct gp_vertex* vertex = &p_scene->vertices[vertex_index];

      vertex->pos_x = ai_position->x;
      vertex->pos_y = ai_position->y;
      vertex->pos_z = ai_position->z;
      vertex->t_u = 0.0;
      vertex->norm_x = ai_normal->x;
      vertex->norm_y = ai_normal->y;
      vertex->norm_z = ai_normal->z;
      vertex->t_v = 0.0;

      vertex_index++;
    }

    for (uint32_t f = 0; f < ai_mesh->mNumFaces; ++f)
    {
      const struct aiFace* ai_face = &ai_mesh->mFaces[f];
      assert(ai_face->mNumIndices == 3);

      struct gp_triangle* triangle = &p_scene->triangles[triangle_index];

      triangle->v0 = ai_face->mIndices[0];
      triangle->v1 = ai_face->mIndices[1];
      triangle->v2 = ai_face->mIndices[2];
      triangle->mat_index = ai_mesh->mMaterialIndex;

      triangle_index++;
    }
  }

  aiReleaseImport(ai_scene);

  return GP_OK;
}

GpResult gp_write_file(
  const uint8_t* data,
  uint32_t data_size_in_bytes,
  const char* file_path)
{
  FILE *file = fopen(file_path, "wb");
  if (file == NULL) {
    return GP_UNABLE_TO_OPEN_FILE;
  }

  fwrite(data, 1, data_size_in_bytes, file);

  const int close_result = fclose(file);
  if (close_result != 0) {
    return GP_UNABLE_TO_CLOSE_FILE;
  }

  return GP_OK;
}

GpResult gp_unload_scene(gp_scene* p_scene)
{
  free(p_scene->vertices);
  free(p_scene->triangles);
  p_scene->vertices = NULL;
  p_scene->vertices_count = 0;
  p_scene->triangles = NULL;
  p_scene->triangles_count = 0;
  return GP_OK;
}

GpResult gp_write_scene(
  const gp_scene* p_scene,
  const char* file_path)
{
  const uint32_t header_size_in_bytes = 16;

  const uint32_t vertex_count = p_scene->vertices_count;
  const uint32_t index_count = p_scene->triangles_count;
  const uint32_t vertex_offset_in_bytes = header_size_in_bytes;
  const uint32_t index_offset_in_bytes =
    vertex_offset_in_bytes + vertex_count * sizeof(gp_vertex);

  const uint32_t buffer_size_in_bytes =
    header_size_in_bytes +
    p_scene->vertices_count * sizeof(gp_vertex) +
    p_scene->triangles_count * sizeof(gp_triangle);

  uint8_t* buffer = malloc(buffer_size_in_bytes);

  memcpy(buffer +  0, &vertex_offset_in_bytes, 4);
  memcpy(buffer +  4, &vertex_count,  4);
  memcpy(buffer +  8, &index_offset_in_bytes,  4);
  memcpy(buffer + 12, &index_count,   4);

  memcpy(
    buffer + vertex_offset_in_bytes,
    p_scene->vertices,
    p_scene->vertices_count * sizeof(gp_vertex)
  );

  memcpy(
    buffer + index_offset_in_bytes,
    p_scene->triangles,
    p_scene->triangles_count * sizeof(gp_triangle)
  );

  const GpResult result = gp_write_file(
    buffer,
    buffer_size_in_bytes,
    file_path
  );

  return result;
}

int main(int argc, const char* argv[])
{
  if (argc == 1 || argc > 2)
  {
    printf("Usage: gatling_preprocess <file_path>\n");
    return EXIT_FAILURE;
  }

  const char* file_path = argv[1];
  const char* file_path_new = "test.gsd";

  gp_scene scene;

  GpResult gp_result;

  gp_result = gp_load_scene(
    &scene,
    file_path
  );
  assert(gp_result == GP_OK);

  gp_result = gp_write_scene(
    &scene,
    file_path_new
  );
  assert(gp_result == GP_OK);

  gp_result = gp_unload_scene(&scene);
  assert(gp_result == GP_OK);

  return EXIT_SUCCESS;
}
