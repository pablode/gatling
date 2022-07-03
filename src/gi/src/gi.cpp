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

#include "bvh.h"
#include "bvh_collapse.h"
#include "bvh_compress.h"
#ifdef GATLING_USE_EMBREE
#include "bvh_embree.h"
#endif
#include "stager.h"
#include "texsys.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>

#include <cgpu.h>
#include <ShaderGen.h>

const uint32_t WORKGROUP_SIZE_X = 32;
const uint32_t WORKGROUP_SIZE_Y = 8;
const float    BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

struct gi_geom_cache
{
  cgpu_buffer                buffer;
  uint64_t                   bvh_node_buf_offset;
  uint64_t                   bvh_node_buf_size;
  uint32_t                   bvh_node_count;
  uint64_t                   face_buf_offset;
  uint64_t                   face_buf_size;
  uint32_t                   face_count;
  uint64_t                   emissive_face_indices_buf_offset;
  uint64_t                   emissive_face_indices_buf_size;
  uint32_t                   emissive_face_count;
  uint64_t                   vertex_buf_offset;
  uint64_t                   vertex_buf_size;
  std::vector<sg::Material*> materials;
};

struct gi_shader_cache
{
  uint32_t                aov_id;
  cgpu_shader             shader;
  cgpu_pipeline           pipeline;
  bool                    nee_enabled;
  bool                    bvh_enabled;
  std::vector<cgpu_image> images_2d;
  std::vector<cgpu_image> images_3d;
  cgpu_buffer             image_mappings;
};

struct gi_material
{
  sg::Material* sg_mat;
};

cgpu_device s_device = { CGPU_INVALID_HANDLE };
cgpu_physical_device_limits s_device_limits;
cgpu_sampler s_tex_sampler = { CGPU_INVALID_HANDLE };
std::unique_ptr<gi::Stager> s_stager;
std::unique_ptr<sg::ShaderGen> s_shaderGen;
std::unique_ptr<gi::TexSys> s_texSys;

cgpu_buffer s_outputBuffer = { CGPU_INVALID_HANDLE };
cgpu_buffer s_outputStagingBuffer = { CGPU_INVALID_HANDLE };
uint32_t s_outputBufferWidth = 0;
uint32_t s_outputBufferHeight = 0;
uint32_t s_sampleOffset = 0;

int giInitialize(const gi_init_params* params)
{
  CgpuResult c_result;

  c_result = cgpu_initialize("gatling", GATLING_VERSION_MAJOR, GATLING_VERSION_MINOR, GATLING_VERSION_PATCH);
  if (c_result != CGPU_OK) return GI_ERROR;

  c_result = cgpu_create_device(&s_device);
  if (c_result != CGPU_OK) return GI_ERROR;

  c_result = cgpu_get_physical_device_limits(s_device, &s_device_limits);
  if (c_result != CGPU_OK) return GI_ERROR;

  c_result = cgpu_create_sampler(s_device,
    CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
    CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
    CGPU_SAMPLER_ADDRESS_MODE_REPEAT,
    &s_tex_sampler
  );
  if (c_result != CGPU_OK) return GI_ERROR;

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

  s_texSys = std::make_unique<gi::TexSys>(s_device, *s_stager);

  return GI_OK;
}

void giTerminate()
{
  cgpu_destroy_buffer(s_device, s_outputStagingBuffer);
  cgpu_destroy_buffer(s_device, s_outputBuffer);
  s_outputBufferWidth = 0;
  s_outputBufferHeight = 0;
  s_texSys->destroy();
  s_texSys.reset();
  s_shaderGen.reset();
  s_stager->free();
  s_stager.reset();
  cgpu_destroy_sampler(s_device, s_tex_sampler);
  cgpu_destroy_device(s_device);
  cgpu_terminate();
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

gi_geom_cache* giCreateGeomCache(const gi_geom_cache_params* params)
{
  printf("creating geom cache\n");

  // Build the BVH.
  gi::bvh::Bvh<8> bvh8;
  gi::bvh::Bvh8c bvh8c;

  // We don't support single-node BVHs, as this requires special handling in the traversal algorithm.
  // Instead, we make sure the number of triangles exceeds the root node's maximum, requiring a child node.
  uint32_t bvh_tri_threshold = std::max(4u, params->bvh_tri_threshold);
  bool bvh_enabled = (params->face_count >= bvh_tri_threshold);

  if (bvh_enabled)
  {
    float node_traversal_cost = 1.0f;
    float face_intersection_cost = 0.3f;

#ifdef GATLING_USE_EMBREE
    gi::bvh::EmbreeBuildParams bvh_params;
    bvh_params.face_batch_size = 1;
    bvh_params.face_count = params->face_count;
    bvh_params.face_intersection_cost = face_intersection_cost;
    bvh_params.faces = params->faces;
    bvh_params.node_traversal_cost = node_traversal_cost;
    bvh_params.vertex_count = params->vertex_count;
    bvh_params.vertices = params->vertices;

    gi::bvh::Bvh2 bvh2 = gi::bvh::build_bvh2_embree(bvh_params);
#else
    gi::bvh::BvhBuildParams bvh_params;
    bvh_params.face_batch_size = 1;
    bvh_params.face_count = params->face_count;
    bvh_params.face_intersection_cost = face_intersection_cost;
    bvh_params.faces = params->faces;
    bvh_params.leaf_max_face_count = 1;
    bvh_params.object_binning_mode = gi::bvh::BvhBinningMode::Fixed;
    bvh_params.object_binning_threshold = 1024;
    bvh_params.object_bin_count = 16;
    bvh_params.spatial_bin_count = 32;
    bvh_params.spatial_split_alpha = 1.0f; // Temporarily disabled.
    bvh_params.vertex_count = params->vertex_count;
    bvh_params.vertices = params->vertices;

    gi::bvh::Bvh2 bvh2 = gi::bvh::build_bvh2(bvh_params);
#endif

    gi::bvh::CollapseParams bvh8_params;
    bvh8_params.max_leaf_size = 3;
    bvh8_params.node_traversal_cost = node_traversal_cost;
    bvh8_params.face_intersection_cost = face_intersection_cost;

    if (!gi::bvh::collapse_bvh2(bvh2, bvh8_params, bvh8))
    {
      return nullptr;
    }

    bvh8c = gi::bvh::compress_bvh8(bvh8);
  }

  uint32_t face_count = bvh_enabled ? bvh8.faces.size() : params->face_count;
  const gi_face* faces = bvh_enabled ? bvh8.faces.data() : params->faces;

  printf("faces: %d\n", face_count);
  printf("vertices: %d\n", params->vertex_count);

  // Build list of emissive faces.
  std::vector<uint16_t> emissive_face_indices;
  if (params->next_event_estimation)
  {
    emissive_face_indices.reserve(128);
    for (uint32_t i = 0; i < face_count; i++)
    {
      const gi_material* mat = params->materials[faces[i].mat_index];
      if (s_shaderGen->isMaterialEmissive(mat->sg_mat))
      {
        emissive_face_indices.push_back(i);
      }
      if (emissive_face_indices.size() == UINT16_MAX)
      {
        fprintf(stderr, "Limit of %d lights for NEE reached; ignoring rest\n", int(UINT16_MAX));
        break;
      }
    }
  }

  // Upload to GPU buffer.
  gi_geom_cache* cache = nullptr;
  cgpu_buffer buffer = { CGPU_INVALID_HANDLE };

  uint64_t buf_size = 0;
  const uint64_t offset_align = s_device_limits.minStorageBufferOffsetAlignment;

  uint64_t bvh_node_buf_size = bvh8c.nodes.size() * sizeof(gi::bvh::Bvh8cNode);
  uint64_t face_buf_size = face_count * sizeof(gi_face);
  uint64_t emissive_face_indices_buf_size = emissive_face_indices.size() * sizeof(uint16_t);
  uint64_t vertex_buf_size = params->vertex_count * sizeof(gi_vertex);

  uint64_t bvh_node_buf_offset = giAlignBuffer(offset_align, bvh_node_buf_size, &buf_size);
  uint64_t face_buf_offset = giAlignBuffer(offset_align, face_buf_size, &buf_size);
  uint64_t emissive_face_indices_buf_offset = giAlignBuffer(offset_align, emissive_face_indices_buf_size, &buf_size);
  uint64_t vertex_buf_offset = giAlignBuffer(offset_align, vertex_buf_size, &buf_size);

  printf("total geom buffer size: %.2fMiB\n", buf_size * BYTES_TO_MIB);
  printf("> %.2fMiB bvh\n", bvh_node_buf_size * BYTES_TO_MIB);
  printf("> %.2fMiB faces\n", face_buf_size * BYTES_TO_MIB);
  printf("> %.2fMiB emissive face indices\n", emissive_face_indices_buf_size * BYTES_TO_MIB);
  printf("> %.2fMiB vertices\n", vertex_buf_size * BYTES_TO_MIB);

  CgpuBufferUsageFlags bufferUsage = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;
  CgpuMemoryPropertyFlags bufferMemProps = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL;

  bool result = cgpu_create_buffer(s_device, bufferUsage, bufferMemProps, buf_size, &buffer) == CGPU_OK;
  if (!result) goto cleanup;

  result = s_stager->stageToBuffer((uint8_t*) bvh8c.nodes.data(), bvh_node_buf_size, buffer, bvh_node_buf_offset);
  if (!result) goto cleanup;
  result = s_stager->stageToBuffer((uint8_t*) faces, face_buf_size, buffer, face_buf_offset);
  if (!result) goto cleanup;
  result = s_stager->stageToBuffer((uint8_t*) emissive_face_indices.data(), emissive_face_indices_buf_size, buffer, emissive_face_indices_buf_offset);
  if (!result) goto cleanup;
  result = s_stager->stageToBuffer((uint8_t*) params->vertices, vertex_buf_size, buffer, vertex_buf_offset);
  if (!result) goto cleanup;

  result = s_stager->flush();
  if (!result) goto cleanup;

  cache = new gi_geom_cache;
  cache->buffer = buffer;
  cache->bvh_node_buf_size = bvh_node_buf_size;
  cache->bvh_node_buf_offset = bvh_node_buf_offset;
  cache->bvh_node_count = bvh8c.nodes.size();
  cache->face_buf_size = face_buf_size;
  cache->face_buf_offset = face_buf_offset;
  cache->face_count = face_count;
  cache->emissive_face_indices_buf_size = emissive_face_indices_buf_size;
  cache->emissive_face_indices_buf_offset = emissive_face_indices_buf_offset;
  cache->emissive_face_count = emissive_face_indices.size();
  cache->vertex_buf_size = vertex_buf_size;
  cache->vertex_buf_offset = vertex_buf_offset;

  // Copy materials.
  cache->materials.resize(params->material_count);
  for (int i = 0; i < cache->materials.size(); i++)
  {
    cache->materials[i] = params->materials[i]->sg_mat;
  }

cleanup:
  if (!cache)
  {
    cgpu_destroy_buffer(s_device, buffer);
  }
  return cache;
}

void giDestroyGeomCache(gi_geom_cache* cache)
{
  cgpu_destroy_buffer(s_device, cache->buffer);
  delete cache;
}

bool giStageImages(const std::vector<sg::TextureResource>& textureResources,
                   std::vector<cgpu_image>& images_2d,
                   std::vector<cgpu_image>& images_3d,
                   cgpu_buffer& imageMappingBuffer)
{
  std::vector<uint16_t> imageMappings;
  if (!s_texSys->loadTextures(textureResources,
                              images_2d,
                              images_3d,
                              imageMappings))
  {
    return false;
  }

  assert(imageMappings.size() < UINT16_MAX);

  uint64_t imageMappingsSize = imageMappings.size() * sizeof(uint16_t);
  bool result = cgpu_create_buffer(
    s_device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    imageMappingsSize,
    &imageMappingBuffer
  ) == CGPU_OK;
  if (!result) return false;

  result = s_stager->stageToBuffer((uint8_t*) imageMappings.data(), imageMappingsSize, imageMappingBuffer, 0);
  if (!result) return false;

  return s_stager->flush();
}

gi_shader_cache* giCreateShaderCache(const gi_shader_cache_params* params)
{
  printf("creating shader cache\n");

  const gi_geom_cache* geom_cache = params->geom_cache;

  float postpone_ratio = 0.2f;
  uint32_t bvh_node_count = geom_cache->bvh_node_count;
  uint32_t bvh_depth = ceilf(log(bvh_node_count) / log(8));
  uint32_t max_bvh_stack_size = (bvh_node_count < 3) ? 1 : (2 * bvh_depth);
  uint32_t max_postponed_tris = int(s_device_limits.subgroupSize * postpone_ratio) - 1;
  uint32_t max_stack_size = max_bvh_stack_size + max_postponed_tris;
  bool bvh_enabled = geom_cache->bvh_node_count > 0;
  bool nee_enabled = geom_cache->emissive_face_count > 0;

  sg::ShaderGen::MainShaderParams shaderParams;
  shaderParams.aovId               = params->aov_id;
  shaderParams.bvh                 = bvh_enabled;
  shaderParams.numThreadsX         = WORKGROUP_SIZE_X;
  shaderParams.numThreadsY         = WORKGROUP_SIZE_Y;
  shaderParams.postponeRatio       = postpone_ratio;
  shaderParams.maxStackSize        = max_stack_size;
  shaderParams.materials           = geom_cache->materials;
  shaderParams.trianglePostponing  = params->triangle_postponing;
  shaderParams.nextEventEstimation = nee_enabled;
  shaderParams.faceCount           = geom_cache->face_count;
  shaderParams.emissiveFaceCount   = geom_cache->emissive_face_count;

  sg::ShaderGen::MainShaderResult mainShader;
  if (!s_shaderGen->generateMainShader(&shaderParams, mainShader))
  {
    return nullptr;
  }

  cgpu_shader shader;
  if (cgpu_create_shader(s_device, mainShader.spv.size(), mainShader.spv.data(), &shader) != CGPU_OK)
  {
    return nullptr;
  }

  cgpu_pipeline pipeline;
  if (cgpu_create_pipeline(s_device, shader, mainShader.entryPoint.c_str(), &pipeline) != CGPU_OK)
  {
    cgpu_destroy_shader(s_device, shader);
    return nullptr;
  }

  std::vector<cgpu_image> images_2d;
  std::vector<cgpu_image> images_3d;
  cgpu_buffer imageMappingBuffer = { CGPU_INVALID_HANDLE };
  if (!giStageImages(mainShader.textureResources, images_2d, images_3d, imageMappingBuffer))
  {
    cgpu_destroy_buffer(s_device, imageMappingBuffer);
    s_texSys->destroyUncachedImages(images_2d);
    s_texSys->destroyUncachedImages(images_3d);
    cgpu_destroy_shader(s_device, shader);
    cgpu_destroy_pipeline(s_device, pipeline);
    return nullptr;
  }

  gi_shader_cache* cache = new gi_shader_cache;
  cache->aov_id = params->aov_id;
  cache->shader = shader;
  cache->pipeline = pipeline;
  cache->nee_enabled = nee_enabled;
  cache->bvh_enabled = bvh_enabled;
  cache->images_2d = std::move(images_2d);
  cache->images_3d = std::move(images_3d);
  cache->image_mappings = imageMappingBuffer;

  return cache;
}

void giDestroyShaderCache(gi_shader_cache* cache)
{
  s_texSys->destroyUncachedImages(cache->images_2d);
  s_texSys->destroyUncachedImages(cache->images_3d);
  cgpu_destroy_buffer(s_device, cache->image_mappings);
  cgpu_destroy_shader(s_device, cache->shader);
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
  CgpuResult c_result;

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

    cgpu_destroy_buffer(s_device, s_outputBuffer);
    cgpu_destroy_buffer(s_device, s_outputStagingBuffer);

    c_result = cgpu_create_buffer(
      s_device,
      CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
      CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
      output_buffer_size,
      &s_outputBuffer
    );
    if (c_result != CGPU_OK) return GI_ERROR;

    c_result = cgpu_create_buffer(
      s_device,
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
      CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
      output_buffer_size,
      &s_outputStagingBuffer
    );
    if (c_result != CGPU_OK) return GI_ERROR;

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

  std::vector<cgpu_buffer_binding> buffers;
  buffers.reserve(16);

  buffers.push_back({ 0, 0, s_outputBuffer, 0, output_buffer_size });
  if (shader_cache->bvh_enabled)
  {
    buffers.push_back({ 1, 0, geom_cache->buffer, geom_cache->bvh_node_buf_offset, geom_cache->bvh_node_buf_size });
  }
  buffers.push_back({ 2, 0, geom_cache->buffer, geom_cache->face_buf_offset, geom_cache->face_buf_size });
  if (shader_cache->nee_enabled)
  {
    buffers.push_back({ 3, 0, geom_cache->buffer, geom_cache->emissive_face_indices_buf_offset, geom_cache->emissive_face_indices_buf_size });
  }
  buffers.push_back({ 4, 0, geom_cache->buffer, geom_cache->vertex_buf_offset, geom_cache->vertex_buf_size });

  uint32_t image_count = shader_cache->images_2d.size() + shader_cache->images_3d.size();

  std::vector<cgpu_image_binding> images;
  images.reserve(image_count);

  cgpu_sampler_binding sampler = { 5, 0, s_tex_sampler };

  if (image_count > 0)
  {
    buffers.push_back({ 6, 0, shader_cache->image_mappings, 0, CGPU_WHOLE_SIZE });
  }

  for (uint32_t i = 0; i < shader_cache->images_2d.size(); i++)
  {
    images.push_back({ 7, i, shader_cache->images_2d[i] });
  }

  for (uint32_t i = 0; i < shader_cache->images_3d.size(); i++)
  {
    images.push_back({ 8, i, shader_cache->images_3d[i] });
  }

  cgpu_bindings bindings= {
    (uint32_t) buffers.size(), buffers.data(),
    (uint32_t) images.size(), images.data(),
    (uint32_t) (image_count ? 1 : 0), &sampler
  };

  // Set up command buffer.
  c_result = cgpu_create_command_buffer(s_device, &command_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_begin_command_buffer(command_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_cmd_update_bindings(
    command_buffer,
    shader_cache->pipeline,
    &bindings
  );
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_cmd_bind_pipeline(command_buffer, shader_cache->pipeline);
  if (c_result != CGPU_OK) goto cleanup;

  // Trace rays.
  c_result = cgpu_cmd_push_constants(
    command_buffer,
    shader_cache->pipeline,
    &push_data
  );
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_cmd_dispatch(
    command_buffer,
    (params->image_width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X,
    (params->image_height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y,
    1
  );
  if (c_result != CGPU_OK) goto cleanup;

  // Copy output buffer to staging buffer.
  cgpu_buffer_memory_barrier barrier;
  barrier.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  barrier.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ;
  barrier.buffer = s_outputBuffer;
  barrier.offset = 0;
  barrier.size = CGPU_WHOLE_SIZE;

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, nullptr,
    1, &barrier,
    0, nullptr
  );
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    s_outputBuffer,
    0,
    s_outputStagingBuffer,
    0,
    output_buffer_size
  );
  if (c_result != CGPU_OK) goto cleanup;

  // Submit command buffer.
  c_result = cgpu_end_command_buffer(command_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_create_fence(s_device, &fence);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_reset_fence(s_device, fence);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_submit_command_buffer(
    s_device,
    command_buffer,
    fence
  );
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_wait_for_fence(s_device, fence);
  if (c_result != CGPU_OK) goto cleanup;

  // Read data from GPU to image.
  uint8_t* mapped_staging_mem;
  c_result = cgpu_map_buffer(
    s_device,
    s_outputStagingBuffer,
    (void**) &mapped_staging_mem
  );
  if (c_result != CGPU_OK) goto cleanup;

  memcpy(rgba_img, mapped_staging_mem, output_buffer_size);

  c_result = cgpu_unmap_buffer(s_device, s_outputStagingBuffer);
  if (c_result != CGPU_OK) goto cleanup;

  // Normalize image for debug AOVs.
  if (shader_cache->aov_id == GI_AOV_ID_DEBUG_BVH_STEPS ||
      shader_cache->aov_id == GI_AOV_ID_DEBUG_TRI_TESTS ||
      shader_cache->aov_id == GI_AOV_ID_DEBUG_BOUNCES)
  {
    const int value_count = pixel_count * color_component_count;

    float max_value = 0.0f;
    for (int i = 0; i < value_count; i++) {
      max_value = std::max(max_value, rgba_img[i]);
    }
    for (int i = 0; i < value_count && max_value > 0.0f; i++) {
      rgba_img[i] /= max_value;
    }
  }

  s_sampleOffset += params->spp;

  result = GI_OK;

cleanup:
  cgpu_destroy_fence(s_device, fence);
  cgpu_destroy_command_buffer(s_device, command_buffer);

  return result;
}
