#include <stdint.h>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <vector>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

typedef enum GpResult {
  GP_OK = 0,
  GP_UNABLE_TO_OPEN_FILE = -1,
  GP_UNABLE_TO_IMPORT_SCENE = -2
} GpResult;

struct gp_vertex
{
  float pos_x;
  float pos_y;
  float pos_z;
  float norm_x;
  float norm_y;
  float norm_z;
  float t_u;
  float t_v;
};

struct gp_scene
{
  std::vector<gp_vertex> vertices;
  std::vector<uint32_t> indices;
};

GpResult gp_load_scene(
  gp_scene& scene,
  const char* p_file_path)
{
  const aiScene* ai_scene = aiImportFile(
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
    std::printf("Unable to import scene: %s\n", error_msg);
    return GP_UNABLE_TO_IMPORT_SCENE;
  }

  scene.vertices.clear();
  scene.indices.clear();

  for (uint32_t m = 0; m < ai_scene->mNumMeshes; ++m)
  {
    const aiMesh* ai_mesh = ai_scene->mMeshes[m];

    for (uint32_t v = 0; v < ai_mesh->mNumVertices; ++v)
    {
      const aiVector3D& ai_position = ai_mesh->mVertices[v];
      const aiVector3D& ai_normal = ai_mesh->mNormals[v];
      const aiVector3D& ai_tex_coords = ai_mesh->mTextureCoords[0][v];
      scene.vertices.push_back({
        .pos_x = ai_position.x,
        .pos_y = ai_position.y,
        .pos_z = ai_position.z,
        .norm_x = ai_normal.x,
        .norm_y = ai_normal.y,
        .norm_z = ai_normal.z,
        .t_u = 0.0,
        .t_v = 0.0
      });
    }

    for (uint32_t f = 0; f < ai_mesh->mNumFaces; ++f)
    {
      const aiFace& ai_face = ai_mesh->mFaces[f];
      assert(ai_face.mNumIndices == 3);
      scene.indices.push_back(ai_face.mIndices[0]);
      scene.indices.push_back(ai_face.mIndices[1]);
      scene.indices.push_back(ai_face.mIndices[2]);
    }
  }

  aiReleaseImport(ai_scene);

  return GP_OK;
}

GpResult gp_write_file(
  const std::vector<uint8_t>& data,
  const char* file_path)
{
  std::ofstream file{
    file_path,
    std::ofstream::out | std::ofstream::binary
  };
  if (!file.is_open()) {
    return GP_UNABLE_TO_OPEN_FILE;
  }
  file.write(
    reinterpret_cast<const char*>(data.data()),
    data.size()
  );
  file.close();

  return GP_OK;
}

GpResult gp_write_scene(
  const gp_scene& scene,
  const char* file_path)
{
  std::vector<uint8_t> buffer;
  buffer.resize(16 +
                scene.vertices.size() * sizeof(gp_vertex) +
                scene.indices.size() * sizeof(uint32_t));

  const uint32_t vertex_offset = 16;
  const uint32_t vertex_count = scene.vertices.size();
  const uint32_t index_offset =
    16 + scene.vertices.size() * sizeof(gp_vertex);
  const uint32_t index_count = scene.indices.size();

  std::memcpy(buffer.data() +  0, &vertex_offset, 4);
  std::memcpy(buffer.data() +  4, &vertex_count,  4);
  std::memcpy(buffer.data() +  8, &index_offset,  4);
  std::memcpy(buffer.data() + 12, &index_count,   4);

  std::memcpy(
    buffer.data() + 16,
    scene.vertices.data(),
    scene.vertices.size() * sizeof(gp_vertex)
  );

  std::memcpy(
    buffer.data() + 16 +
      (scene.vertices.size() * sizeof(gp_vertex)),
    scene.indices.data(),
    scene.indices.size() * sizeof(uint32_t)
  );

  const GpResult gp_result =
    gp_write_file(buffer, file_path);

  return gp_result;
}

int main(int argc, const char* argv[])
{
  if (argc == 1 || argc > 2) {
    std::printf("Usage: gatling_preprocess <file_path>\n");
    return EXIT_FAILURE;
  }

  const char* file_path = argv[1];
  const char* file_path_new = "test.gsd";

  gp_scene scene;

  GpResult gp_result;

  gp_result = gp_load_scene(
    scene,
    file_path
  );
  assert(gp_result == GP_OK);

  gp_result = gp_write_scene(
    scene,
    file_path_new
  );
  assert(gp_result == GP_OK);

  return EXIT_SUCCESS;
}
