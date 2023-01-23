//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "gi.h"

#include "Stager.h"
#include "texsys.h"
#include "turbo.h"
#include "asset_reader.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <atomic>
#include <assert.h>

#include <cgpu.h>
#include <gml.h>
#include <sg/ShaderGen.h>

using namespace gi;

const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

struct gi_geom_cache
{
  std::vector<cgpu_blas> blases;
  cgpu_buffer            buffer;
  uint32_t               emissive_face_count;
  uint64_t               emissive_face_indices_buf_offset;
  uint64_t               emissive_face_indices_buf_size;
  uint64_t               face_buf_offset;
  uint64_t               face_buf_size;
  uint32_t               face_count;
  cgpu_tlas              tlas;
  uint64_t               vertex_buf_offset;
  uint64_t               vertex_buf_size;
};

struct gi_shader_cache
{
  uint32_t                        aov_id;
  std::vector<cgpu_shader>        hitShaders;
  std::vector<cgpu_image>         images_2d;
  std::vector<cgpu_image>         images_3d;
  std::vector<const gi_material*> materials;
  cgpu_shader                     missShader;
  cgpu_pipeline                   pipeline;
  cgpu_shader                     rgenShader;
};

struct gi_material
{
  sg::Material* sg_mat;
};

struct gi_mesh
{
  std::vector<gi_face> faces;
  std::vector<gi_vertex> vertices;
  const gi_material* material;
};

cgpu_device s_device = { CGPU_INVALID_HANDLE };
cgpu_physical_device_features s_device_features;
cgpu_physical_device_properties s_device_properties;
cgpu_sampler s_tex_sampler = { CGPU_INVALID_HANDLE };
std::unique_ptr<gi::Stager> s_stager;
std::unique_ptr<sg::ShaderGen> s_shaderGen;
std::unique_ptr<GiMmapAssetReader> s_mmapAssetReader;
std::unique_ptr<GiAggregateAssetReader> s_aggregateAssetReader;
std::unique_ptr<gi::TexSys> s_texSys;

cgpu_buffer s_outputBuffer = { CGPU_INVALID_HANDLE };
cgpu_buffer s_outputStagingBuffer = { CGPU_INVALID_HANDLE };
uint32_t s_outputBufferWidth = 0;
uint32_t s_outputBufferHeight = 0;
uint32_t s_sampleOffset = 0;

int giInitialize(const gi_init_params* params)
{
  if (!cgpu_initialize("gatling", GATLING_VERSION_MAJOR, GATLING_VERSION_MINOR, GATLING_VERSION_PATCH))
    return GI_ERROR;

  if (!cgpu_create_device(&s_device))
    return GI_ERROR;

  if (!cgpu_get_physical_device_features(s_device, &s_device_features))
    return GI_ERROR;

  if (!cgpu_get_physical_device_properties(s_device, &s_device_properties))
    return GI_ERROR;

  if (!cgpu_create_sampler(s_device,
                           CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
                           CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
                           CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
                           &s_tex_sampler))
  {
    return GI_ERROR;
  }

  s_stager = std::make_unique<gi::Stager>(s_device);
  if (!s_stager->allocate())
  {
    return GI_ERROR;
  }

  sg::ShaderGen::InitParams sgParams;
  sgParams.resourcePath = params->resource_path;
  sgParams.shaderPath = params->shader_path;
  sgParams.mtlxLibPath = params->mtlx_lib_path;
  sgParams.mdlLibPath = params->mdl_lib_path;

  s_shaderGen = std::make_unique<sg::ShaderGen>();
  if (!s_shaderGen->init(sgParams))
  {
    return GI_ERROR;
  }

  s_mmapAssetReader = std::make_unique<GiMmapAssetReader>();
  s_aggregateAssetReader = std::make_unique<GiAggregateAssetReader>();
  s_aggregateAssetReader->addAssetReader(s_mmapAssetReader.get());

  s_texSys = std::make_unique<gi::TexSys>(s_device, *s_aggregateAssetReader, *s_stager);

  return GI_OK;
}

void giTerminate()
{
  s_aggregateAssetReader.reset();
  s_mmapAssetReader.reset();
  cgpu_destroy_buffer(s_device, s_outputStagingBuffer);
  s_outputStagingBuffer.handle = CGPU_INVALID_HANDLE;
  cgpu_destroy_buffer(s_device, s_outputBuffer);
  s_outputBuffer.handle = CGPU_INVALID_HANDLE;
  s_outputBufferWidth = 0;
  s_outputBufferHeight = 0;
  if (s_texSys)
  {
    s_texSys->destroy();
    s_texSys.reset();
  }
  s_shaderGen.reset();
  if (s_stager)
  {
    s_stager->free();
    s_stager.reset();
  }
  cgpu_destroy_sampler(s_device, s_tex_sampler);
  cgpu_destroy_device(s_device);
  cgpu_terminate();
}

void giRegisterAssetReader(GiAssetReader* reader)
{
  s_aggregateAssetReader->addAssetReader(reader);
}

gi_material* giCreateMaterialFromMtlx(const char* doc_str)
{
  sg::Material* sg_mat = s_shaderGen->createMaterialFromMtlx(doc_str);
  if (!sg_mat)
  {
    return nullptr;
  }

  gi_material* mat = new gi_material;
  mat->sg_mat = sg_mat;
  return mat;
}

gi_material* giCreateMaterialFromMdlFile(const char* file_path, const char* sub_identifier)
{
  sg::Material* sg_mat = s_shaderGen->createMaterialFromMdlFile(file_path, sub_identifier);
  if (!sg_mat)
  {
    return nullptr;
  }

  gi_material* mat = new gi_material;
  mat->sg_mat = sg_mat;
  return mat;
}

void giDestroyMaterial(gi_material* mat)
{
  s_shaderGen->destroyMaterial(mat->sg_mat);
  delete mat;
}

uint64_t giAlignBuffer(uint64_t alignment, uint64_t buffer_size, uint64_t* total_size)
{
  if (buffer_size == 0)
  {
    return *total_size;
  }

  const uint64_t offset = ((*total_size) + alignment - 1) / alignment * alignment;

  (*total_size) = offset + buffer_size;

  return offset;
}

gi_mesh* giCreateMesh(const gi_mesh_desc* desc)
{
  gi_mesh* mesh = new gi_mesh;
  mesh->faces = std::vector<gi_face>(&desc->faces[0], &desc->faces[desc->face_count]);
  mesh->vertices = std::vector<gi_vertex>(&desc->vertices[0], &desc->vertices[desc->vertex_count]);
  mesh->material = desc->material;
  return mesh;
}

bool _giBuildGeometryStructures(const gi_geom_cache_params* params,
                                std::vector<cgpu_blas>& blases,
                                std::vector<cgpu_blas_instance>& blas_instances,
                                std::vector<gi_vertex>& allVertices,
                                std::vector<gi_face>& allFaces,
                                std::vector<uint32_t>& emissive_face_indices)
{
  for (uint32_t m = 0; m < params->mesh_count; m++)
  {
    const gi_mesh* mesh = params->meshes[m];

    if (mesh->faces.empty())
    {
      continue;
    }

    uint32_t faceIndexOffset = allFaces.size();
    uint32_t vertexIndexOffset = allVertices.size();

    // Vertices
    std::vector<cgpu_vertex> vertices;
    vertices.resize(mesh->vertices.size());
    allVertices.reserve(allVertices.size() + mesh->vertices.size());

    for (uint32_t i = 0; i < vertices.size(); i++)
    {
      vertices[i].x = mesh->vertices[i].pos[0];
      vertices[i].y = mesh->vertices[i].pos[1];
      vertices[i].z = mesh->vertices[i].pos[2];

      allVertices.push_back(mesh->vertices[i]);
    }

    // Indices
    std::vector<uint32_t> indices;
    indices.reserve(mesh->faces.size() * 3);
    allFaces.reserve(allFaces.size() + mesh->faces.size());

    for (uint32_t i = 0; i < mesh->faces.size(); i++)
    {
      const auto* face = &mesh->faces[i];
      indices.push_back(face->v_i[0]);
      indices.push_back(face->v_i[1]);
      indices.push_back(face->v_i[2]);

      gi_face new_face;
      new_face.v_i[0] = vertexIndexOffset + face->v_i[0];
      new_face.v_i[1] = vertexIndexOffset + face->v_i[1];
      new_face.v_i[2] = vertexIndexOffset + face->v_i[2];
      allFaces.push_back(new_face);
    }

    // BLAS
    cgpu_blas blas;
    if (!cgpu_create_blas(s_device, (uint32_t)vertices.size(), vertices.data(), (uint32_t)indices.size(), indices.data(), &blas))
    {
      goto fail_cleanup;
    }

    blases.push_back(blas);

    // FIXME: find a better solution
    uint32_t materialIndex = UINT32_MAX;
    gi_shader_cache* shader_cache = params->shader_cache;
    for (uint32_t i = 0; i < shader_cache->materials.size(); i++)
    {
      if (shader_cache->materials[i] == mesh->material)
      {
        materialIndex = i;
        break;
      }
    }
    if (materialIndex == UINT32_MAX)
    {
      goto fail_cleanup;
    }

    // Instance
    cgpu_blas_instance blas_instance = { 0 };
    blas_instance.as = blas;
    blas_instance.faceIndexOffset = faceIndexOffset;
    blas_instance.hitShaderIndex = materialIndex;
    float transform[3][4] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f
    };
    memcpy(blas_instance.transform, transform, sizeof(float) * 12);

    blas_instances.push_back(blas_instance);
  }
  return true;

fail_cleanup:
  assert(false);
  for (cgpu_blas blas : blases)
  {
    cgpu_destroy_blas(s_device, blas);
  }
  return false;
}

gi_geom_cache* giCreateGeomCache(const gi_geom_cache_params* params)
{
  gi_geom_cache* cache = nullptr;

  printf("creating geom cache\n");
  printf("mesh count: %d\n", params->mesh_count);

  // Build HW ASes and vertex, index buffers.
  cgpu_buffer buffer = { CGPU_INVALID_HANDLE };
  cgpu_tlas tlas = { CGPU_INVALID_HANDLE };
  std::vector<cgpu_blas> blases;
  std::vector<cgpu_blas_instance> blas_instances;
  std::vector<gi_vertex> allVertices;
  std::vector<gi_face> allFaces;
  std::vector<uint32_t> emissive_face_indices; // FIXME: reintroduce

  if (!_giBuildGeometryStructures(params, blases, blas_instances, allVertices, allFaces, emissive_face_indices))
    goto cleanup;

  if (!cgpu_create_tlas(s_device, blas_instances.size(), blas_instances.data(), &tlas))
    goto cleanup;

  // Upload vertex, index buffers to single GPU buffer.
  uint64_t face_buf_size;
  uint64_t emissive_face_indices_buf_size;
  uint64_t vertex_buf_size;
  uint64_t face_buf_offset;
  uint64_t emissive_face_indices_buf_offset;
  uint64_t vertex_buf_offset;
  {
    uint64_t buf_size = 0;
    const uint64_t offset_align = s_device_properties.minStorageBufferOffsetAlignment;

    face_buf_size = allFaces.size() * sizeof(gi_face);
    emissive_face_indices_buf_size = emissive_face_indices.size() * sizeof(uint32_t);
    vertex_buf_size = allVertices.size() * sizeof(gi_vertex);

    face_buf_offset = giAlignBuffer(offset_align, face_buf_size, &buf_size);
    emissive_face_indices_buf_offset = giAlignBuffer(offset_align, emissive_face_indices_buf_size, &buf_size);
    vertex_buf_offset = giAlignBuffer(offset_align, vertex_buf_size, &buf_size);

    printf("total geom buffer size: %.2fMiB\n", buf_size * BYTES_TO_MIB);
    printf("> %.2fMiB faces\n", face_buf_size * BYTES_TO_MIB);
    printf("> %.2fMiB emissive face indices\n", emissive_face_indices_buf_size * BYTES_TO_MIB);
    printf("> %.2fMiB vertices\n", vertex_buf_size * BYTES_TO_MIB);

    CgpuBufferUsageFlags bufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
    CgpuMemoryPropertyFlags bufferMemProps = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL;

    if (!cgpu_create_buffer(s_device, bufferUsage, bufferMemProps, buf_size, &buffer))
      goto cleanup;

    if (!s_stager->stageToBuffer((uint8_t*)allFaces.data(), face_buf_size, buffer, face_buf_offset))
      goto cleanup;
    if (!s_stager->stageToBuffer((uint8_t*)emissive_face_indices.data(), emissive_face_indices_buf_size, buffer, emissive_face_indices_buf_offset))
      goto cleanup;
    if (!s_stager->stageToBuffer((uint8_t*)allVertices.data(), vertex_buf_size, buffer, vertex_buf_offset))
      goto cleanup;

    if (!s_stager->flush())
      goto cleanup;
  }

  // Fill cache struct.
  cache = new gi_geom_cache;
  cache->tlas = tlas;
  cache->blases = blases;
  cache->buffer = buffer;
  cache->face_buf_size = face_buf_size;
  cache->face_buf_offset = face_buf_offset;
  cache->face_count = allFaces.size();
  cache->emissive_face_indices_buf_size = emissive_face_indices_buf_size;
  cache->emissive_face_indices_buf_offset = emissive_face_indices_buf_offset;
  cache->emissive_face_count = (uint32_t) emissive_face_indices.size();
  cache->vertex_buf_size = vertex_buf_size;
  cache->vertex_buf_offset = vertex_buf_offset;

cleanup:
  if (!cache)
  {
    assert(false);
    if (buffer.handle != CGPU_INVALID_HANDLE)
    {
      cgpu_destroy_buffer(s_device, buffer);
    }
    if (tlas.handle != CGPU_INVALID_HANDLE)
    {
      cgpu_destroy_tlas(s_device, tlas);
    }
    for (cgpu_blas blas : blases)
    {
      cgpu_destroy_blas(s_device, blas);
    }
  }
  return cache;
}

void giDestroyGeomCache(gi_geom_cache* cache)
{
  for (cgpu_blas blas : cache->blases)
  {
    cgpu_destroy_blas(s_device, blas);
  }
  cgpu_destroy_tlas(s_device, cache->tlas);
  cgpu_destroy_buffer(s_device, cache->buffer);
  delete cache;
}

gi_shader_cache* giCreateShaderCache(const gi_shader_cache_params* params)
{
  bool clockCyclesAov = params->aov_id == GI_AOV_ID_DEBUG_CLOCK_CYCLES;

  if (clockCyclesAov && !s_device_features.shaderClock)
  {
    fprintf(stderr, "error: unsupported AOV - device feature missing\n");
    return nullptr;
  }

  printf("creating shader cache\n");
  printf("material count: %d\n", params->material_count);

  gi_shader_cache* cache = nullptr;
  cgpu_pipeline pipeline = { CGPU_INVALID_HANDLE };
  cgpu_shader rgenShader = { CGPU_INVALID_HANDLE };
  cgpu_shader missShader = { CGPU_INVALID_HANDLE };
  std::vector<cgpu_shader> hitShaders;
  std::vector<cgpu_image> images_2d;
  std::vector<cgpu_image> images_3d;
  std::vector<sg::TextureResource> textureResources;

  // Create per-material closest-hit shaders.
  //
  // This is done in multiple phases: first, GLSL is generated from MDL, and
  // texture information is extracted. The information is then used to generated
  // the descriptor sets for the pipeline. Lastly, the GLSL is stiched, #defines
  // are added, and the code is compiled to SPIR-V.
  {
    std::vector<sg::Material*> materials;
    materials.resize(params->material_count);
    for (int i = 0; i < params->material_count; i++)
    {
      materials[i] = params->materials[i]->sg_mat;
    }

    // 1. Generate GLSL from MDL
    std::vector<sg::ShaderGen::ClosestHitShaderIntermediates> hitShaderIntermediates;
    hitShaderIntermediates.resize(params->material_count);

    std::atomic_bool threadWorkFailed = false;
#pragma omp parallel for
    for (int i = 0; i < hitShaderIntermediates.size(); i++)
    {
      const gi_material* mat = params->materials[i];

      sg::ShaderGen::ClosestHitShaderIntermediates intermediates;
      if (s_shaderGen->generateClosestHitIntermediates(mat->sg_mat, intermediates))
      {
        hitShaderIntermediates[i] = intermediates;
      }
      else
      {
        threadWorkFailed = true;
      }
    }
    if (threadWorkFailed)
    {
      goto cleanup;
    }

    // 2. Sum up texture resources & calculate per-material index offsets.
    std::vector<uint32_t> textureIndexOffsets;
    for (const auto& intermediates : hitShaderIntermediates)
    {
      textureIndexOffsets.push_back(textureResources.size());

      for (const auto& tr : intermediates.textureResources)
      {
        textureResources.push_back(tr);
      }
    }

    // 3. Generate final GLSL hit shaders, and compile them to SPIR-V.
    hitShaders.resize(hitShaderIntermediates.size(), { CGPU_INVALID_HANDLE });
    threadWorkFailed = false;
#pragma omp parallel for
    for (int i = 0; i < hitShaderIntermediates.size(); i++)
    {
      const sg::ShaderGen::ClosestHitShaderIntermediates& intermediates = hitShaderIntermediates[i];

      sg::ShaderGen::ClosestHitShaderParams hitParams;
      hitParams.aovId = params->aov_id;
      hitParams.baseFileName = "rt_main.chit";
      hitParams.mdlGeneratedGlsl = intermediates.mdlGeneratedGlsl;
      hitParams.textureResources = &textureResources;
      hitParams.textureIndexOffset = textureIndexOffsets[i];

      std::vector<uint8_t> spv;
      if (s_shaderGen->generateClosestHitSpirv(hitParams, spv))
      {
        cgpu_shader closestHitShader;
        if (cgpu_create_shader(s_device, spv.size(), spv.data(), CGPU_SHADER_STAGE_CLOSEST_HIT, &closestHitShader))
        {
          hitShaders[i] = closestHitShader;
        }
        else
        {
          threadWorkFailed = true;
        }
      }
      else
      {
        threadWorkFailed = true;
      }
    }
    if (threadWorkFailed)
    {
      goto cleanup;
    }
  }

  // Create ray generation and miss shader.
  {
    std::vector<uint8_t> rgenSpirv;
    sg::ShaderGen::RaygenShaderParams rgenParams;
    rgenParams.shaderClockExts = clockCyclesAov;
    rgenParams.textureResources = &textureResources;

    std::vector<uint8_t> missSpirv;
    sg::ShaderGen::MissShaderParams missParams;
    missParams.textureResources = &textureResources;

    if (!s_shaderGen->generateRgenSpirv("rt_main.rgen", rgenParams, rgenSpirv) ||
        !s_shaderGen->generateMissSpirv("rt_main.miss", missParams, missSpirv))
    {
      goto cleanup;
    }

    if (!cgpu_create_shader(s_device, rgenSpirv.size(), rgenSpirv.data(), CGPU_SHADER_STAGE_RAYGEN, &rgenShader))
    {
      goto cleanup;
    }

    if (!cgpu_create_shader(s_device, missSpirv.size(), missSpirv.data(), CGPU_SHADER_STAGE_MISS, &missShader))
    {
      goto cleanup;
    }
  }

  // Upload textures.
  if (textureResources.size() > 0 && !s_texSys->loadTextures(textureResources, images_2d, images_3d))
  {
    goto cleanup;
  }

  // Create RT pipeline.
  {
    cgpu_rt_pipeline_desc pipeline_desc = {0};
    pipeline_desc.rgen_shader = rgenShader;
    pipeline_desc.miss_shader_count = 1;
    pipeline_desc.miss_shaders = &missShader;
    pipeline_desc.hit_shader_count = hitShaders.size();
    pipeline_desc.hit_shaders = hitShaders.data();

    if (!cgpu_create_rt_pipeline(s_device, &pipeline_desc, &pipeline))
    {
      goto cleanup;
    }
  }

  cache = new gi_shader_cache;
  cache->aov_id = params->aov_id;
  cache->hitShaders = std::move(hitShaders);
  cache->images_2d = std::move(images_2d);
  cache->images_3d = std::move(images_3d);
  cache->materials.resize(params->material_count);
  for (int i = 0; i < params->material_count; i++)
  {
    cache->materials[i] = params->materials[i];
  }
  cache->missShader = missShader;
  cache->pipeline = pipeline;
  cache->rgenShader = rgenShader;

cleanup:
  if (!cache)
  {
    assert(false);
    s_texSys->destroyUncachedImages(images_2d);
    s_texSys->destroyUncachedImages(images_3d);
    if (rgenShader.handle != CGPU_INVALID_HANDLE)
    {
      cgpu_destroy_shader(s_device, rgenShader);
    }
    if (missShader.handle != CGPU_INVALID_HANDLE)
    {
      cgpu_destroy_shader(s_device, missShader);
    }
    for (cgpu_shader shader : hitShaders)
    {
      cgpu_destroy_shader(s_device, shader);
    }
    if (pipeline.handle != CGPU_INVALID_HANDLE)
    {
      cgpu_destroy_pipeline(s_device, pipeline);
    }
  }
  return cache;
}

void giDestroyShaderCache(gi_shader_cache* cache)
{
  s_texSys->destroyUncachedImages(cache->images_2d);
  s_texSys->destroyUncachedImages(cache->images_3d);
  cgpu_destroy_shader(s_device, cache->rgenShader);
  cgpu_destroy_shader(s_device, cache->missShader);
  for (cgpu_shader shader : cache->hitShaders)
  {
    cgpu_destroy_shader(s_device, shader);
  }
  cgpu_destroy_pipeline(s_device, cache->pipeline);
  delete cache;
}

void giInvalidateFramebuffer()
{
  s_sampleOffset = 0;
}

int giRender(const gi_render_params* params, float* rgba_img)
{
  const gi_geom_cache* geom_cache = params->geom_cache;
  const gi_shader_cache* shader_cache = params->shader_cache;

  // Init state for goto error handling.
  int result = GI_ERROR;

  cgpu_command_buffer command_buffer = { CGPU_INVALID_HANDLE };
  cgpu_fence fence = { CGPU_INVALID_HANDLE };

  // Set up output buffer.
  int color_component_count = 4;
  int pixel_count = params->image_width * params->image_height;
  uint64_t output_buffer_size = pixel_count * color_component_count * sizeof(float);

  bool reallocOutputBuffer = s_outputBufferWidth != params->image_width ||
                             s_outputBufferHeight != params->image_height;

  if (reallocOutputBuffer)
  {
    printf("recreating output buffer with size %dx%d (%.2fMiB)\n", params->image_width,
      params->image_height, output_buffer_size * BYTES_TO_MIB);

    if (s_outputBuffer.handle != CGPU_INVALID_HANDLE)
      cgpu_destroy_buffer(s_device, s_outputBuffer);
    if (s_outputStagingBuffer.handle != CGPU_INVALID_HANDLE)
      cgpu_destroy_buffer(s_device, s_outputStagingBuffer);

    if (!cgpu_create_buffer(s_device,
                            CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
                            CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
                            output_buffer_size,
                            &s_outputBuffer))
    {
      return GI_ERROR;
    }

    if (!cgpu_create_buffer(s_device,
                            CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
                            CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
                            output_buffer_size,
                            &s_outputStagingBuffer))
    {
      return GI_ERROR;
    }

    s_outputBufferWidth = params->image_width;
    s_outputBufferHeight = params->image_height;
  }

  // Set up GPU data.
  gml_vec3 cam_forward, cam_up;
  gml_vec3_normalize(params->camera->forward, cam_forward);
  gml_vec3_normalize(params->camera->up, cam_up);

  float push_data[] = {
    params->camera->position[0], params->camera->position[1], params->camera->position[2], // float3
    *((float*)&params->image_width),                                                       // uint
    cam_forward[0], cam_forward[1], cam_forward[2],                                        // float3
    *((float*)&params->image_height),                                                      // uint
    cam_up[0], cam_up[1], cam_up[2],                                                       // float3
    params->camera->vfov,                                                                  // float
    params->bg_color[0], params->bg_color[1], params->bg_color[2], params->bg_color[3],    // float4
    *((float*)&params->spp),                                                               // uint
    *((float*)&params->max_bounces),                                                       // uint
    params->max_sample_value,                                                              // float
    *((float*)&params->rr_bounce_offset),                                                  // uint
    params->rr_inv_min_term_prob,                                                          // float
    *((float*)&s_sampleOffset)                                                             // uint
  };
  uint32_t push_size = sizeof(push_data);

  std::vector<cgpu_buffer_binding> buffers;
  buffers.reserve(16);

  buffers.push_back({ 0, 0, s_outputBuffer, 0, output_buffer_size });
  buffers.push_back({ 1, 0, geom_cache->buffer, geom_cache->face_buf_offset, geom_cache->face_buf_size });
  if (false/*FIXME: shader_cache->nee_enabled*/)
  {
    buffers.push_back({ 2, 0, geom_cache->buffer, geom_cache->emissive_face_indices_buf_offset, geom_cache->emissive_face_indices_buf_size });
  }
  buffers.push_back({ 3, 0, geom_cache->buffer, geom_cache->vertex_buf_offset, geom_cache->vertex_buf_size });

  size_t image_count = shader_cache->images_2d.size() + shader_cache->images_3d.size();

  std::vector<cgpu_image_binding> images;
  images.reserve(image_count);

  cgpu_sampler_binding sampler = { 4, 0, s_tex_sampler };

  for (uint32_t i = 0; i < shader_cache->images_2d.size(); i++)
  {
    images.push_back({ 5, i, shader_cache->images_2d[i] });
  }

  for (uint32_t i = 0; i < shader_cache->images_3d.size(); i++)
  {
    images.push_back({ 6, i, shader_cache->images_3d[i] });
  }

  cgpu_tlas_binding as = { 7, 0, geom_cache->tlas };

  cgpu_bindings bindings = {0};
  bindings.buffer_count = (uint32_t) buffers.size();
  bindings.p_buffers = buffers.data();
  bindings.image_count = (uint32_t) images.size();
  bindings.p_images = images.data();
  bindings.sampler_count = image_count ? 1 : 0;
  bindings.p_samplers = &sampler;
  bindings.tlas_count = 1;
  bindings.p_tlases = &as;

  // Set up command buffer.
  if (!cgpu_create_command_buffer(s_device, &command_buffer))
    goto cleanup;

  if (!cgpu_begin_command_buffer(command_buffer))
    goto cleanup;

  if (!cgpu_cmd_transition_shader_image_layouts(command_buffer, shader_cache->rgenShader, images.size(), images.data()))
    goto cleanup;

  if (!cgpu_cmd_update_bindings(command_buffer, shader_cache->pipeline, &bindings))
    goto cleanup;

  if (!cgpu_cmd_bind_pipeline(command_buffer, shader_cache->pipeline))
    goto cleanup;

  // Trace rays.
  if (!cgpu_cmd_push_constants(command_buffer, shader_cache->pipeline, CGPU_SHADER_STAGE_RAYGEN | CGPU_SHADER_STAGE_MISS | CGPU_SHADER_STAGE_CLOSEST_HIT, push_size, &push_data))
    goto cleanup;

  if (!cgpu_cmd_trace_rays(command_buffer, shader_cache->pipeline, params->image_width, params->image_height))
    goto cleanup;

  // Copy output buffer to staging buffer.
  cgpu_buffer_memory_barrier barrier;
  barrier.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  barrier.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ;
  barrier.buffer = s_outputBuffer;
  barrier.offset = 0;
  barrier.size = CGPU_WHOLE_SIZE;

  if (!cgpu_cmd_pipeline_barrier(
      command_buffer,
      0, nullptr,
      1, &barrier,
      0, nullptr))
  {
    goto cleanup;
  }

  if (!cgpu_cmd_copy_buffer(
      command_buffer,
      s_outputBuffer,
      0,
      s_outputStagingBuffer,
      0,
      output_buffer_size))
  {
    goto cleanup;
  }

  // Submit command buffer.
  if (!cgpu_end_command_buffer(command_buffer))
    goto cleanup;

  if (!cgpu_create_fence(s_device, &fence))
    goto cleanup;

  if (!cgpu_reset_fence(s_device, fence))
    goto cleanup;

  if (!cgpu_submit_command_buffer(
      s_device,
      command_buffer,
      fence))
  {
    goto cleanup;
  }

  // Now is a good time to flush buffered messages (on Windows).
  fflush(stdout);

  if (!cgpu_wait_for_fence(s_device, fence))
    goto cleanup;

  // Read data from GPU to image.
  uint8_t* mapped_staging_mem;
  if (!cgpu_map_buffer(s_device, s_outputStagingBuffer, (void**) &mapped_staging_mem))
  {
    goto cleanup;
  }

  memcpy(rgba_img, mapped_staging_mem, output_buffer_size);

  if (!cgpu_unmap_buffer(s_device, s_outputStagingBuffer))
    goto cleanup;

  // Visualize red channel as heatmap for debug AOVs.
  if (shader_cache->aov_id == GI_AOV_ID_DEBUG_BOUNCES ||
      shader_cache->aov_id == GI_AOV_ID_DEBUG_CLOCK_CYCLES)
  {
    int value_count = pixel_count * color_component_count;

    float max_value = 0.0f;
    for (int i = 0; i < value_count; i += 4) {
      max_value = std::max(max_value, rgba_img[i]);
    }
    for (int i = 0; i < value_count && max_value > 0.0f; i += 4) {
      int val_index = std::min(int((rgba_img[i] / max_value) * 255.0), 255);
      rgba_img[i + 0] = (float) gi::TURBO_SRGB_FLOATS[val_index][0];
      rgba_img[i + 1] = (float) gi::TURBO_SRGB_FLOATS[val_index][1];
      rgba_img[i + 2] = (float) gi::TURBO_SRGB_FLOATS[val_index][2];
      rgba_img[i + 3] = 255;
    }
  }

  s_sampleOffset += params->spp;

  result = GI_OK;

cleanup:
  cgpu_destroy_fence(s_device, fence);
  cgpu_destroy_command_buffer(s_device, command_buffer);

  return result;
}
