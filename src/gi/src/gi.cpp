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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>

#include <cgpu.h>
#include <ShaderGen.h>

const uint32_t WORKGROUP_SIZE_X = 16;
const uint32_t WORKGROUP_SIZE_Y = 16;

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
  std::vector<cgpu_image> images;
};

struct gi_material
{
  sg::Material* sg_mat;
};

cgpu_device s_device;
cgpu_physical_device_limits s_device_limits;
cgpu_sampler s_tex_sampler;
std::unique_ptr<gi::Stager> s_stager;
std::unique_ptr<sg::ShaderGen> s_shaderGen;

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
  sgParams.mtlxlibPath = params->mtlxlib_path;
  sgParams.mtlxmdlPath = params->mtlxmdl_path;

  s_shaderGen = std::make_unique<sg::ShaderGen>();
  if (!s_shaderGen->init(sgParams))
  {
    return GI_ERROR;
  }

  return GI_OK;
}

void giTerminate()
{
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

static uint64_t giAlignBuffer(uint64_t alignment,
                                uint64_t buffer_size,
                                uint64_t* total_size)
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

  // Build list of emissive faces.
  std::vector<uint32_t> emissive_face_indices;
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
    }
  }

  // Upload to GPU buffer.
  gi_geom_cache* cache = nullptr;
  cgpu_buffer buffer = { CGPU_INVALID_HANDLE };
  cgpu_buffer staging_buffer = { CGPU_INVALID_HANDLE };
  cgpu_command_buffer command_buffer = { CGPU_INVALID_HANDLE };
  cgpu_fence fence = { CGPU_INVALID_HANDLE };

  uint64_t buf_size = 0;
  const uint64_t offset_align = s_device_limits.minStorageBufferOffsetAlignment;

  uint64_t bvh_node_buf_size = bvh8c.nodes.size() * sizeof(gi::bvh::Bvh8cNode);
  uint64_t face_buf_size = face_count * sizeof(gi_face);
  uint64_t emissive_face_indices_buf_size = emissive_face_indices.size() * sizeof(uint32_t);
  uint64_t vertex_buf_size = params->vertex_count * sizeof(gi_vertex);

  uint64_t bvh_node_buf_offset = giAlignBuffer(offset_align, bvh_node_buf_size, &buf_size);
  uint64_t face_buf_offset = giAlignBuffer(offset_align, face_buf_size, &buf_size);
  uint64_t emissive_face_indices_buf_offset = giAlignBuffer(offset_align, emissive_face_indices_buf_size, &buf_size);
  uint64_t vertex_buf_offset = giAlignBuffer(offset_align, vertex_buf_size, &buf_size);

  CgpuResult c_result = cgpu_create_buffer(
    s_device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    buf_size,
    &buffer
  );
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_create_buffer(
    s_device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
    buf_size,
    &staging_buffer
  );
  if (c_result != CGPU_OK) goto cleanup;

  uint8_t* mapped_staging_mem;
  c_result = cgpu_map_buffer(
    s_device,
    staging_buffer,
    (void**) &mapped_staging_mem
  );
  if (c_result != CGPU_OK) goto cleanup;

  // from memcpy docs: "If either dest or src is an invalid or null pointer, the behavior is undefined, even if count is zero."
  if (bvh8c.nodes.size() > 0) {
    memcpy(&mapped_staging_mem[bvh_node_buf_offset], bvh8c.nodes.data(), bvh_node_buf_size);
  }
  if (face_count > 0) {
    memcpy(&mapped_staging_mem[face_buf_offset], faces, face_buf_size);
  }
  if (emissive_face_indices.size() > 0) {
    memcpy(&mapped_staging_mem[emissive_face_indices_buf_offset], emissive_face_indices.data(), emissive_face_indices_buf_size);
  }
  if (params->vertex_count > 0) {
    memcpy(&mapped_staging_mem[vertex_buf_offset], params->vertices, vertex_buf_size);
  }

  c_result = cgpu_unmap_buffer(s_device, staging_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_create_command_buffer(s_device, &command_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_begin_command_buffer(command_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    staging_buffer,
    0,
    buffer,
    0,
    buf_size
  );
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_end_command_buffer(command_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_create_fence(s_device, &fence);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_reset_fence(s_device, fence);

  c_result = cgpu_submit_command_buffer(
    s_device,
    command_buffer,
    fence
  );
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_wait_for_fence(s_device, fence);
  if (c_result != CGPU_OK) goto cleanup;

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
  cgpu_destroy_fence(s_device, fence);
  cgpu_destroy_command_buffer(s_device, command_buffer);
  cgpu_destroy_buffer(s_device, staging_buffer);
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

gi_shader_cache* giCreateShaderCache(const gi_shader_cache_params* params)
{
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

  uint32_t texCount = mainShader.textureResources.size();

  std::vector<cgpu_image> images;
  images.reserve(texCount);

  for (int i = 0; i < texCount; i++)
  {
    cgpu_image image = { CGPU_INVALID_HANDLE };

    CgpuResult c_result = cgpu_create_image(s_device,
      1, 1, CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM,
      CGPU_IMAGE_USAGE_FLAG_SAMPLED,
      &image
    );
    assert(c_result == CGPU_OK);

    uint8_t black[4] = { 0, 0, 0, 0 };
    s_stager->stageToImage(black, 4, image);

    images.push_back(image);
  }

  s_stager->flush();

  gi_shader_cache* cache = new gi_shader_cache;
  cache->aov_id = params->aov_id;
  cache->shader = shader;
  cache->pipeline = pipeline;
  cache->nee_enabled = nee_enabled;
  cache->bvh_enabled = bvh_enabled;
  cache->images = std::move(images);

  return cache;
}

void giDestroyShaderCache(gi_shader_cache* cache)
{
  for (cgpu_image image : cache->images)
  {
    cgpu_destroy_image(s_device, image);
  }
  cgpu_destroy_shader(s_device, cache->shader);
  cgpu_destroy_pipeline(s_device, cache->pipeline);
  delete cache;
}

int giRender(const gi_render_params* params,
             float* rgba_img)
{
  const gi_geom_cache* geom_cache = params->geom_cache;
  const gi_shader_cache* shader_cache = params->shader_cache;

  // Init state for goto error handling.
  int result = GI_ERROR;
  CgpuResult c_result;

  cgpu_buffer output_buffer = { CGPU_INVALID_HANDLE };
  cgpu_buffer staging_buffer = { CGPU_INVALID_HANDLE };
  cgpu_command_buffer command_buffer = { CGPU_INVALID_HANDLE };
  cgpu_fence fence = { CGPU_INVALID_HANDLE };

  // Set up buffers.
  const int COLOR_COMPONENT_COUNT = 4;
  const int pixel_count = params->image_width * params->image_height;
  const uint64_t buffer_size = pixel_count * COLOR_COMPONENT_COUNT * sizeof(float);

  c_result = cgpu_create_buffer(
    s_device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
    buffer_size,
    &staging_buffer
  );
  if (c_result != CGPU_OK)
  {
    return GI_ERROR;
  }

  c_result = cgpu_create_buffer(
    s_device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    buffer_size,
    &output_buffer
  );
  if (c_result != CGPU_OK)
  {
    cgpu_destroy_buffer(s_device, staging_buffer);
    return GI_ERROR;
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
    params->rr_inv_min_term_prob                                                           // float
  };

  std::vector<cgpu_buffer_binding> buffers;
  buffers.reserve(16);

  buffers.push_back({ 0, 0, output_buffer, 0, CGPU_WHOLE_SIZE });
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

  uint32_t texCount = shader_cache->images.size();

  std::vector<cgpu_image_binding> images;
  images.reserve(texCount);

  for (uint32_t i = 0; i < texCount; i++)
  {
    images.push_back({ 5, i, shader_cache->images[i] });
  }

  cgpu_sampler_binding sampler = { 6, 0, s_tex_sampler };

  cgpu_bindings bindings= {
    (uint32_t) buffers.size(), buffers.data(),
    (uint32_t) images.size(), images.data(),
    1, &sampler
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
  barrier.buffer = output_buffer;
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
    output_buffer,
    0,
    staging_buffer,
    0,
    buffer_size
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
    staging_buffer,
    (void**) &mapped_staging_mem
  );
  if (c_result != CGPU_OK) goto cleanup;

  memcpy(rgba_img, mapped_staging_mem, buffer_size);

  c_result = cgpu_unmap_buffer(
    s_device,
    staging_buffer
  );
  if (c_result != CGPU_OK) goto cleanup;

  // Normalize image for debug AOVs.
  if (shader_cache->aov_id == GI_AOV_ID_DEBUG_BVH_STEPS ||
      shader_cache->aov_id == GI_AOV_ID_DEBUG_TRI_TESTS)
  {
    const int value_count = pixel_count * COLOR_COMPONENT_COUNT;

    float max_value = 0.0f;
    for (int i = 0; i < value_count; i++) {
      max_value = std::max(max_value, rgba_img[i]);
    }
    for (int i = 0; i < value_count && max_value > 0.0f; i++) {
      rgba_img[i] /= max_value;
    }
  }

  result = GI_OK;

cleanup:
  cgpu_destroy_fence(s_device, fence);
  cgpu_destroy_command_buffer(s_device, command_buffer);
  cgpu_destroy_buffer(s_device, staging_buffer);
  cgpu_destroy_buffer(s_device, output_buffer);

  return result;
}
