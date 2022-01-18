#include "gi.h"

#include "bvh.h"
#include "bvh_collapse.h"
#include "bvh_compress.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>

#include <cgpu.h>
#include <shadergen.h>

const uint32_t WORKGROUP_SIZE_X = 16;
const uint32_t WORKGROUP_SIZE_Y = 16;

struct gi_geom_cache
{
  uint32_t                   bvh_node_count;
  cgpu_buffer                buffer;
  uint64_t                   node_buf_offset;
  uint64_t                   node_buf_size;
  uint64_t                   face_buf_offset;
  uint64_t                   face_buf_size;
  uint64_t                   vertex_buf_offset;
  uint64_t                   vertex_buf_size;
  uint32_t                   material_count;
  std::vector<sg::Material*> materials;
};

struct gi_shader_cache
{
  cgpu_shader shader;
  std::string shader_entry_point;
};

struct gi_material
{
  sg::Material* sg_mat;
};

cgpu_device s_device;
cgpu_physical_device_limits s_device_limits;
std::unique_ptr<sg::ShaderGen> s_shaderGen;

int giInitialize(const gi_init_params* params)
{
  if (cgpu_initialize("gatling", GATLING_VERSION_MAJOR, GATLING_VERSION_MINOR, GATLING_VERSION_PATCH) != CGPU_OK)
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

  /* Set up GPU device. */
  CgpuResult c_result;

  uint32_t device_count;
  c_result = cgpu_get_device_count(&device_count);
  if (c_result != CGPU_OK) return GI_ERROR;

  if (device_count == 0)
  {
    fprintf(stderr, "No device found!\n");
    return GI_ERROR;
  }

  c_result = cgpu_create_device(0, &s_device);
  if (c_result != CGPU_OK) return GI_ERROR;

  c_result = cgpu_get_physical_device_limits(s_device, &s_device_limits);
  if (c_result != CGPU_OK) return GI_ERROR;

  return GI_OK;
}

void giTerminate()
{
  s_shaderGen.reset();
  cgpu_destroy_device(s_device);
  cgpu_terminate();
}

gi_material* giCreateMaterialFromMtlx(const char* doc_str)
{
  sg::Material* sg_mat = s_shaderGen->createMaterialFromMtlx(doc_str);
  if (!sg_mat)
  {
    return NULL;
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
  const uint64_t offset = ((*total_size) + alignment - 1) / alignment * alignment;

  (*total_size) = offset + buffer_size;

  return offset;
}

gi_geom_cache* giCreateGeomCache(const gi_geom_cache_params* params)
{
  /* We don't support too few faces since this would lead to the root node
   * being a leaf, requiring special handling in the traversal algorithm. */
  if (params->face_count <= 3)
  {
    return NULL;
  }

  /* Build BVH. */
  gi_bvh_params bvh_params;
  bvh_params.face_batch_size          = 1;
  bvh_params.face_count               = params->face_count;
  bvh_params.face_intersection_cost   = 1.2f;
  bvh_params.faces                    = params->faces;
  bvh_params.leaf_max_face_count      = 1;
  bvh_params.object_binning_mode      = GI_BVH_BINNING_MODE_FIXED;
  bvh_params.object_binning_threshold = 1024;
  bvh_params.object_bin_count         = 16;
  bvh_params.spatial_bin_count        = 32;
  bvh_params.spatial_reserve_factor   = 1.25f;
  bvh_params.spatial_split_alpha      = 1.0f; /* Temporarily disabled. */
  bvh_params.vertex_count             = params->vertex_count;
  bvh_params.vertices                 = params->vertices;

  gi_bvh bvh;
  gi_bvh_build(&bvh_params, &bvh);

  gi_bvhc_params bvhc_params;
  bvhc_params.bvh                    = &bvh;
  bvhc_params.max_leaf_size          = 3;
  bvhc_params.node_traversal_cost    = 1.0f;
  bvhc_params.face_intersection_cost = 0.3f;

  gi_bvhc bvhc;
  gi_bvh_collapse(&bvhc_params, &bvhc);

  gi_free_bvh(&bvh);

  gi_bvhcc bvhcc;
  gi_bvh_compress(&bvhc, &bvhcc);

  /* Upload to GPU buffer. */
  gi_geom_cache* cache = NULL;
  cgpu_buffer buffer = { CGPU_INVALID_HANDLE };
  cgpu_buffer staging_buffer = { CGPU_INVALID_HANDLE };
  cgpu_command_buffer command_buffer = { CGPU_INVALID_HANDLE };
  cgpu_fence fence = { CGPU_INVALID_HANDLE };

  uint64_t buf_size = 0;
  const uint64_t offset_align = s_device_limits.minStorageBufferOffsetAlignment;

  uint64_t node_buf_size = bvhcc.node_count * sizeof(gi_bvhcc_node);
  uint64_t face_buf_size = params->face_count * sizeof(gi_face);
  uint64_t vertex_buf_size = params->vertex_count * sizeof(gi_vertex);

  uint64_t node_buf_offset = giAlignBuffer(offset_align, node_buf_size, &buf_size);
  uint64_t face_buf_offset = giAlignBuffer(offset_align, face_buf_size, &buf_size);
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
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
    buf_size,
    &staging_buffer
  );
  if (c_result != CGPU_OK) goto cleanup;

  uint8_t* mapped_staging_mem;
  c_result = cgpu_map_buffer(
    s_device,
    staging_buffer,
    0,
    buf_size,
    (void**) &mapped_staging_mem
  );
  if (c_result != CGPU_OK) goto cleanup;

  memcpy(&mapped_staging_mem[node_buf_offset], bvhcc.nodes, node_buf_size);
  memcpy(&mapped_staging_mem[face_buf_offset], bvhc.faces, face_buf_size);
  memcpy(&mapped_staging_mem[vertex_buf_offset], params->vertices, vertex_buf_size);

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
  cache->bvh_node_count = bvhcc.node_count;
  cache->buffer = buffer;
  cache->node_buf_size = node_buf_size;
  cache->face_buf_size = face_buf_size;
  cache->vertex_buf_size = vertex_buf_size;
  cache->node_buf_offset = node_buf_offset;
  cache->face_buf_offset = face_buf_offset;
  cache->vertex_buf_offset = vertex_buf_offset;

  /* Copy materials. */
  cache->material_count = params->material_count;
  cache->materials.resize(cache->material_count);
  for (int i = 0; i < cache->material_count; i++)
  {
    cache->materials[i] = params->materials[i]->sg_mat;
  }

cleanup:
  gi_free_bvhc(&bvhc);
  gi_free_bvhcc(&bvhcc);

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
  uint32_t node_count = params->geom_cache->bvh_node_count;
  uint32_t max_stack_size = (node_count < 3) ? 1 : (log(node_count) * 2 / log(8));

  sg::ShaderGen::MainShaderParams shaderParams;
  shaderParams.numThreadsX      = WORKGROUP_SIZE_X;
  shaderParams.numThreadsY      = WORKGROUP_SIZE_Y;
  shaderParams.maxStackSize     = max_stack_size;
  shaderParams.spp              = params->spp;
  shaderParams.maxBounces       = params->max_bounces;
  shaderParams.maxSampleValue   = params->max_sample_value;
  shaderParams.rrBounceOffset   = params->rr_bounce_offset;
  shaderParams.rrInvMinTermProb = params->rr_inv_min_term_prob;
  shaderParams.bgColor[0]       = params->bg_color[0];
  shaderParams.bgColor[1]       = params->bg_color[1];
  shaderParams.bgColor[2]       = params->bg_color[2];
  shaderParams.bgColor[3]       = params->bg_color[3];
  shaderParams.materials        = params->geom_cache->materials;

  std::vector<uint8_t> spv;
  std::string shader_entry_point;
  if (!s_shaderGen->generateMainShader(&shaderParams, spv, shader_entry_point))
  {
    return NULL;
  }

  cgpu_shader shader;
  if (cgpu_create_shader(s_device, spv.size(), spv.data(), &shader) != CGPU_OK)
  {
    return NULL;
  }

  gi_shader_cache* cache = new gi_shader_cache;
  cache->shader_entry_point = shader_entry_point;
  cache->shader = shader;

  return cache;
}

void giDestroyShaderCache(gi_shader_cache* cache)
{
  cgpu_destroy_shader(s_device, cache->shader);
  delete cache;
}

int giRender(const gi_render_params* params,
             float* rgba_img)
{
  int result = GI_ERROR;
  CgpuResult c_result;

  cgpu_buffer output_buffer = { CGPU_INVALID_HANDLE };
  cgpu_buffer staging_buffer = { CGPU_INVALID_HANDLE };
  cgpu_pipeline pipeline = { CGPU_INVALID_HANDLE };
  cgpu_command_buffer command_buffer = { CGPU_INVALID_HANDLE };
  cgpu_fence fence = { CGPU_INVALID_HANDLE };

  /* Set up buffers. */
  const int COLOR_COMPONENT_COUNT = 4;
  const uint64_t buffer_size = params->image_width * params->image_height * sizeof(float) * COLOR_COMPONENT_COUNT;

  c_result = cgpu_create_buffer(
    s_device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
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

  /* Set up pipeline. */
  gml_vec3 cam_forward, cam_up;
  gml_vec3_normalize(params->camera->forward, cam_forward);
  gml_vec3_normalize(params->camera->up, cam_up);

  float push_data[] = {
    params->camera->position[0],
    params->camera->position[1],
    params->camera->position[2],
    *((float*)&params->image_width),
    cam_forward[0],
    cam_forward[1],
    cam_forward[2],
    *((float*)&params->image_height),
    cam_up[0],
    cam_up[1],
    cam_up[2],
    params->camera->vfov
  };
  uint32_t push_data_size = sizeof(push_data);

  cgpu_shader_resource_buffer buffers[] = {
    { 0,              output_buffer,                                     0,                     CGPU_WHOLE_SIZE },
    { 1, params->geom_cache->buffer,   params->geom_cache->node_buf_offset,   params->geom_cache->node_buf_size },
    { 2, params->geom_cache->buffer,   params->geom_cache->face_buf_offset,   params->geom_cache->face_buf_size },
    { 3, params->geom_cache->buffer, params->geom_cache->vertex_buf_offset, params->geom_cache->vertex_buf_size },
  };
  const uint32_t buffer_count = sizeof(buffers) / sizeof(buffers[0]);

  const uint32_t image_count = 0;
  const cgpu_shader_resource_image* images = NULL;

  c_result = cgpu_create_pipeline(
    s_device,
    buffer_count,
    buffers,
    image_count,
    images,
    params->shader_cache->shader,
    params->shader_cache->shader_entry_point.c_str(),
    push_data_size,
    &pipeline
  );
  if (c_result != CGPU_OK) goto cleanup;

  /* Set up command buffer. */
  c_result = cgpu_create_command_buffer(s_device, &command_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_begin_command_buffer(command_buffer);
  if (c_result != CGPU_OK) goto cleanup;

  c_result = cgpu_cmd_bind_pipeline(command_buffer, pipeline);
  if (c_result != CGPU_OK) goto cleanup;

  /* Trace rays. */
  c_result = cgpu_cmd_push_constants(
    command_buffer,
    pipeline,
    push_data_size,
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

  /* Copy output buffer to staging buffer. */
  cgpu_buffer_memory_barrier barrier;
  barrier.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  barrier.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ;
  barrier.buffer = output_buffer;
  barrier.offset = 0;
  barrier.size = CGPU_WHOLE_SIZE;

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &barrier,
    0, NULL
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

  /* Submit command buffer. */
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

  /* Read data from GPU to image. */
  uint8_t* mapped_staging_mem;
  c_result = cgpu_map_buffer(
    s_device,
    staging_buffer,
    0,
    buffer_size,
    (void**) &mapped_staging_mem
  );
  if (c_result != CGPU_OK) goto cleanup;

  memcpy(rgba_img, mapped_staging_mem, buffer_size);

  c_result = cgpu_unmap_buffer(
    s_device,
    staging_buffer
  );
  if (c_result != CGPU_OK) goto cleanup;

  result = GI_OK;

cleanup:
  cgpu_destroy_fence(s_device, fence);
  cgpu_destroy_command_buffer(s_device, command_buffer);
  cgpu_destroy_pipeline(s_device, pipeline);
  cgpu_destroy_buffer(s_device, staging_buffer);
  cgpu_destroy_buffer(s_device, output_buffer);

  return result;
}
