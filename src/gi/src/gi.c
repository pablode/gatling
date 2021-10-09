#include "gi.h"

#include "bvh.h"
#include "bvh_collapse.h"
#include "bvh_compress.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include <cgpu.h>
#include <shadergen.h>

struct gi_material
{
  struct SgMaterial* sg_mat;
};

struct gi_scene_cache
{
  struct gi_bvhcc     bvhcc;
  uint32_t            face_count;
  struct gi_face*     faces;
  struct gi_material* materials;
  uint32_t            material_count;
  uint32_t            vertex_count;
  struct gi_vertex*   vertices;
};

int giInitialize(const char* shader_path,
                 const char* mtlxlib_path)
{
  if (cgpu_initialize("gatling", GATLING_VERSION_MAJOR, GATLING_VERSION_MINOR, GATLING_VERSION_PATCH) != CGPU_OK)
  {
    return GI_ERROR;
  }
  if (!sgInitialize(shader_path, mtlxlib_path))
  {
    return GI_ERROR;
  }
  return GI_OK;
}

void giTerminate()
{
  sgTerminate();
  cgpu_terminate();
}

struct gi_material* giCreateMaterialFromMtlx(const char* doc_str)
{
  struct SgMaterial* sg_mat = sgCreateMaterialFromMtlx(doc_str);
  if (!sg_mat)
  {
    return NULL;
  }

  struct gi_material* mat = malloc(sizeof(struct gi_material));
  mat->sg_mat = sg_mat;
  return mat;
}

void giDestroyMaterial(struct gi_material* mat)
{
  sgDestroyMaterial(mat->sg_mat);
  free(mat);
}

int giCreateSceneCache(struct gi_scene_cache** cache)
{
  *cache = malloc(sizeof(struct gi_scene_cache));

  return (*cache == NULL) ? GI_ERROR : GI_OK;
}

void giDestroySceneCache(struct gi_scene_cache* cache)
{
  gi_free_bvhcc(&cache->bvhcc);
  free(cache->materials);
  free(cache->vertices);
  free(cache->faces);
  free(cache);
}

int giPreprocess(const struct gi_preprocess_params* params,
                 struct gi_scene_cache* scene_cache)
{
  /* We don't support too few faces since this would lead to the root node
   * being a leaf, requiring special handling in the traversal algorithm. */
  if (params->face_count <= 3)
  {
    return GI_ERROR;
  }

  /* Build BVH. */
  struct gi_bvh bvh;
  const struct gi_bvh_params bvh_params = {
    .face_batch_size          = 1,
    .face_count               = params->face_count,
    .face_intersection_cost   = 1.2f,
    .faces                    = params->faces,
    .leaf_max_face_count      = 1,
    .object_binning_mode      = GI_BVH_BINNING_MODE_FIXED,
    .object_binning_threshold = 1024,
    .object_bin_count         = 16,
    .spatial_bin_count        = 32,
    .spatial_reserve_factor   = 1.25f,
    .spatial_split_alpha      = 1.0f,
    .vertex_count             = params->vertex_count,
    .vertices                 = params->vertices
  };

  gi_bvh_build(&bvh_params, &bvh);

  struct gi_bvhc bvhc;
  struct gi_bvhc_params bvhc_params = {
    .bvh                    = &bvh,
    .max_leaf_size          = 3,
    .node_traversal_cost    = 1.0f,
    .face_intersection_cost = 0.3f
  };

  gi_bvh_collapse(&bvhc_params, &bvhc);
  gi_free_bvh(&bvh);

  /* Copy vertices, materials and new faces. */
  scene_cache->face_count = bvhc.face_count;
  scene_cache->faces = malloc(scene_cache->face_count * sizeof(struct gi_face));
  memcpy(scene_cache->faces, bvhc.faces, bvhc.face_count * sizeof(struct gi_face));

  scene_cache->vertex_count = params->vertex_count;
  scene_cache->vertices = malloc(scene_cache->vertex_count * sizeof(struct gi_vertex));
  memcpy(scene_cache->vertices, params->vertices, params->vertex_count * sizeof(struct gi_vertex));

  scene_cache->material_count = params->material_count;
  scene_cache->materials = malloc(scene_cache->material_count * sizeof(struct gi_material));
  memcpy(scene_cache->materials, params->materials, params->material_count * sizeof(struct gi_material));

  gi_bvh_compress(&bvhc, &scene_cache->bvhcc);
  gi_free_bvhc(&bvhc);

  return GI_OK;
}

static uint64_t gi_align_buffer(uint64_t alignment,
                                uint64_t buffer_size,
                                uint64_t* total_size)
{
  const uint64_t offset = ((*total_size) + alignment - 1) / alignment * alignment;

  (*total_size) = offset + buffer_size;

  return offset;
}

#define GI_CGPU_VERIFY(result)                                                                       \
  do {                                                                                               \
    if (result != CGPU_OK) {                                                                         \
      fprintf(stderr, "Gatling encountered a fatal CGPU error at line %d: %d\n", __LINE__, result);  \
      return GI_ERROR;                                                                               \
    }                                                                                                \
  } while (0)

int giRender(const struct gi_render_params* params,
             float* rgba_img)
{
  /* Set up device. */
  CgpuResult c_result;

  uint32_t device_count;
  c_result = cgpu_get_device_count(&device_count);
  GI_CGPU_VERIFY(c_result);

  if (device_count == 0) {
    fprintf(stderr, "No device found!\n");
    return GI_ERROR;
  }

  cgpu_device device;
  c_result = cgpu_create_device(0, &device);
  GI_CGPU_VERIFY(c_result);

  cgpu_physical_device_limits device_limits;
  c_result = cgpu_get_physical_device_limits(device, &device_limits);
  GI_CGPU_VERIFY(c_result);

  /* Set up GPU buffers. */
  uint64_t device_buf_size = 0;
  const uint64_t offset_align = device_limits.minStorageBufferOffsetAlignment;

  uint64_t node_buf_size = params->scene_cache->bvhcc.node_count * sizeof(struct gi_bvhcc_node);
  uint64_t face_buf_size = params->scene_cache->face_count * sizeof(struct gi_face);
  uint64_t vertex_buf_size = params->scene_cache->vertex_count * sizeof(struct gi_vertex);
  uint64_t material_buf_size = params->scene_cache->material_count * sizeof(struct gi_material);

  const uint64_t new_node_buf_offset = gi_align_buffer(offset_align, node_buf_size, &device_buf_size);
  const uint64_t new_face_buf_offset = gi_align_buffer(offset_align, face_buf_size, &device_buf_size);
  const uint64_t new_vertex_buf_offset = gi_align_buffer(offset_align, vertex_buf_size, &device_buf_size);
  const uint64_t new_material_buf_offset = gi_align_buffer(offset_align, material_buf_size, &device_buf_size);

  const int COLOR_COMPONENT_COUNT = 4;
  const uint64_t output_buffer_size = params->image_width * params->image_height * sizeof(float) * COLOR_COMPONENT_COUNT;
  const uint64_t staging_buffer_size = output_buffer_size > device_buf_size ? output_buffer_size : device_buf_size;

  cgpu_buffer input_buffer;
  cgpu_buffer staging_buffer;
  cgpu_buffer output_buffer;

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    device_buf_size,
    &input_buffer
  );
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
    staging_buffer_size,
    &staging_buffer
  );
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER | CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    output_buffer_size,
    &output_buffer
  );
  GI_CGPU_VERIFY(c_result);

  uint8_t* mapped_staging_mem;
  c_result = cgpu_map_buffer(
    device,
    staging_buffer,
    0,
    device_buf_size,
    (void*)&mapped_staging_mem
  );
  GI_CGPU_VERIFY(c_result);

  memcpy(&mapped_staging_mem[new_node_buf_offset], params->scene_cache->bvhcc.nodes, node_buf_size);
  memcpy(&mapped_staging_mem[new_face_buf_offset], params->scene_cache->faces, face_buf_size);
  memcpy(&mapped_staging_mem[new_vertex_buf_offset], params->scene_cache->vertices, vertex_buf_size);
  memcpy(&mapped_staging_mem[new_material_buf_offset], params->scene_cache->materials, material_buf_size);

  c_result = cgpu_unmap_buffer(
    device,
    staging_buffer
  );
  GI_CGPU_VERIFY(c_result);

  /* We always work on tiles. */
  uint32_t workgroup_size_x = 16;
  uint32_t workgroup_size_y = 16;

  /* Set up pipeline. */
  gml_vec3 cam_forward, cam_up;
  gml_vec3_normalize(params->camera->forward, cam_forward);
  gml_vec3_normalize(params->camera->up, cam_up);

  float push_data[] = {
    params->camera->position[0],
    params->camera->position[1],
    params->camera->position[2],
    params->image_width,
    cam_forward[0],
    cam_forward[1],
    cam_forward[2],
    params->image_height,
    cam_up[0],
    cam_up[1],
    cam_up[2],
    params->camera->vfov
  };
  uint32_t push_size = sizeof(push_data);

  cgpu_pipeline pipeline;
  {
    uint32_t node_size = sizeof(struct gi_bvhcc_node);
    uint32_t node_count = node_buf_size / node_size;
    uint32_t max_stack_size = (node_count < 3) ? 1 : (log(node_count) * 2 / log(8));

    struct SgMainShaderParams shaderParams = {
      .num_threads_x = workgroup_size_x,
      .num_threads_y = workgroup_size_y,
      .max_stack_size = max_stack_size,
      .spp = params->spp,
      .max_bounces = params->max_bounces,
      .rr_bounce_offset = params->rr_bounce_offset,
      .rr_inv_min_term_prob = params->rr_inv_min_term_prob,
    };

    uint32_t spv_size;
    uint32_t* spv;
    const char* entry_point;
    if (!sgGenerateMainShader(&shaderParams, &spv_size, &spv, &entry_point))
    {
      return GI_ERROR;
    }

    cgpu_shader shader;
    c_result = cgpu_create_shader(
      device,
      spv_size,
      spv,
      &shader
    );
    free(spv);
    GI_CGPU_VERIFY(c_result);

    cgpu_shader_resource_buffer sr_buffers[] = {
      { 0,       output_buffer,                       0,   CGPU_WHOLE_SIZE },
      { 1,        input_buffer,     new_node_buf_offset,     node_buf_size },
      { 2,        input_buffer,     new_face_buf_offset,     face_buf_size },
      { 3,        input_buffer,   new_vertex_buf_offset,   vertex_buf_size },
      { 4,        input_buffer, new_material_buf_offset, material_buf_size },
    };
    const uint32_t sr_buffer_count = sizeof(sr_buffers) / sizeof(sr_buffers[0]);

    const uint32_t sr_image_count = 0;
    const cgpu_shader_resource_image* sr_images = NULL;
    const uint32_t push_constants_size = 0;

    c_result = cgpu_create_pipeline(
      device,
      sr_buffer_count,
      sr_buffers,
      sr_image_count,
      sr_images,
      shader,
      entry_point,
      push_size,
      &pipeline
    );
    GI_CGPU_VERIFY(c_result);

    c_result = cgpu_destroy_shader(device, shader);
    GI_CGPU_VERIFY(c_result);
  }

  /* Set up command buffer. */
  cgpu_command_buffer command_buffer;
  c_result = cgpu_create_command_buffer(device, &command_buffer);
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_begin_command_buffer(command_buffer);
  GI_CGPU_VERIFY(c_result);

  /* Copy staging buffer to input buffer. */
  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    staging_buffer,
    0,
    input_buffer,
    0,
    device_buf_size
  );
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &(cgpu_buffer_memory_barrier) {
      .src_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE,
      .dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ,
      .buffer = input_buffer,
      .offset = 0,
      .size = CGPU_WHOLE_SIZE
    },
    0, NULL
  );
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_cmd_bind_pipeline(command_buffer, pipeline);
  GI_CGPU_VERIFY(c_result);

  /* Trace rays. */
  c_result = cgpu_cmd_push_constants(
    command_buffer,
    pipeline,
    push_size,
    push_data
  );
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_cmd_dispatch(
    command_buffer,
    (params->image_width + workgroup_size_x - 1) / workgroup_size_x,
    (params->image_height + workgroup_size_y - 1) / workgroup_size_y,
    1
  );
  GI_CGPU_VERIFY(c_result);

  /* Copy output buffer to staging buffer. */
  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &(cgpu_buffer_memory_barrier) {
      .src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE,
      .dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ,
      .buffer = output_buffer,
      .offset = 0,
      .size = CGPU_WHOLE_SIZE
    },
    0, NULL
  );
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    output_buffer,
    0,
    staging_buffer,
    0,
    output_buffer_size
  );
  GI_CGPU_VERIFY(c_result);

  /* Submit command buffer. */
  c_result = cgpu_end_command_buffer(command_buffer);
  GI_CGPU_VERIFY(c_result);

  cgpu_fence fence;
  c_result = cgpu_create_fence(device, &fence);
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_reset_fence(device, fence);
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_submit_command_buffer(
    device,
    command_buffer,
    fence
  );
  GI_CGPU_VERIFY(c_result);

  c_result = cgpu_wait_for_fence(device, fence);
  GI_CGPU_VERIFY(c_result);

  /* Read data from GPU to image. */
  c_result = cgpu_map_buffer(
    device,
    staging_buffer,
    0,
    output_buffer_size,
    (void**) &mapped_staging_mem
  );
  GI_CGPU_VERIFY(c_result);

  memcpy(
    rgba_img,
    mapped_staging_mem,
    output_buffer_size
  );

  c_result = cgpu_unmap_buffer(
    device,
    staging_buffer
  );
  GI_CGPU_VERIFY(c_result);

  /* Clean up. */
  c_result = cgpu_destroy_fence(device, fence);
  GI_CGPU_VERIFY(c_result);
  c_result = cgpu_destroy_command_buffer(device, command_buffer);
  GI_CGPU_VERIFY(c_result);
  c_result = cgpu_destroy_pipeline(device, pipeline);
  GI_CGPU_VERIFY(c_result);
  c_result = cgpu_destroy_buffer(device, input_buffer);
  GI_CGPU_VERIFY(c_result);
  c_result = cgpu_destroy_buffer(device, staging_buffer);
  GI_CGPU_VERIFY(c_result);
  c_result = cgpu_destroy_buffer(device, output_buffer);
  GI_CGPU_VERIFY(c_result);
  c_result = cgpu_destroy_device(device);
  GI_CGPU_VERIFY(c_result);

  return GI_OK;
}
