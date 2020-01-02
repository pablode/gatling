#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <cgpu.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_WIDTH 3840
#define IMAGE_HEIGHT 2160
#define NUM_SAMPLES 1

typedef enum GatlingResult {
  GATLING_OK = 0,
  GATLING_FAIL_UNABLE_TO_WRITE_OUTPUT_IMAGE = -1,
  GATLING_FAIL_UNABLE_TO_OPEN_FILE = -2,
  GATLING_FAIL_UNABLE_TO_CLOSE_FILE = -3,
  GATLING_FAIL_UNABLE_TO_CREATE_SHADER = -4,
  GATLING_FAIL_UNABLE_TO_CREATE_PIPELINE = -5,
  GATLING_FAIL_UNABLE_TO_DESTROY_SHADER = -6,
  GATLING_FAIL_UNABLE_TO_DESTROY_PIPELINE = -7
} GatlingResult;

GatlingResult gatling_save_img(
  const float* data,
  size_t data_size_in_floats,
  const char* file_path)
{
  uint8_t* temp_data = malloc(data_size_in_floats);

  for (size_t i = 0; i < data_size_in_floats; ++i)
  {
    int32_t color = (int32_t) (data[i]  * 255.0f);
    if (color < 0)   { color = 0;   }
    if (color > 255) { color = 255; }
    temp_data[i] = (uint8_t) color;
  }

  stbi_flip_vertically_on_write(true);

  const uint32_t num_components = 4;
  const int result = stbi_write_png(
    file_path,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    num_components,
    temp_data,
    IMAGE_WIDTH * num_components
  );

  free(temp_data);

  if (!result) {
    return GATLING_FAIL_UNABLE_TO_WRITE_OUTPUT_IMAGE;
  }

  return GATLING_OK;
}

GatlingResult gatling_read_file(
  const char* file_path,
  uint8_t** data,
  size_t* data_size)
{
  FILE *file = fopen(file_path, "rb");
  if (file == NULL) {
    return GATLING_FAIL_UNABLE_TO_OPEN_FILE;
  }

  fseeko(file, 0, SEEK_END);
  *data_size = ftello(file);
  fseeko(file, 0, SEEK_SET);

  *data = malloc(*data_size);
  fread(*data, 1, *data_size, file);

  const int close_result = fclose(file);
  if (close_result != 0) {
    return GATLING_FAIL_UNABLE_TO_CLOSE_FILE;
  }

  return GATLING_OK;
}

typedef struct gatling_pipeline {
  cgpu_pipeline pipeline;
  cgpu_shader shader;
} gatling_pipeline;

GatlingResult gatling_create_pipeline(
  cgpu_device device,
  const char *shader_file_path,
  cgpu_shader_resource_buffer *shader_resource_buffers,
  size_t num_shader_resource_buffers,
  gatling_pipeline *pipeline)
{
  uint32_t* data;
  size_t data_size;

  GatlingResult g_result = gatling_read_file(
    shader_file_path,
    (uint8_t**) &data,
    &data_size
  );
  if (g_result != GATLING_OK) {
    return g_result;
  }

  CgpuResult c_result = cgpu_create_shader(
    device,
    data_size,
    data,
    &pipeline->shader
  );
  if (c_result != CGPU_OK) {
    return GATLING_FAIL_UNABLE_TO_CREATE_PIPELINE;
  }

  free(data);

  c_result = cgpu_create_pipeline(
    device,
    num_shader_resource_buffers,
    shader_resource_buffers,
    0,
    NULL,
    pipeline->shader,
    "main",
    &pipeline->pipeline
  );
  if (c_result != CGPU_OK) {
    return GATLING_FAIL_UNABLE_TO_CREATE_PIPELINE;
  }

  return GATLING_OK;
}

GatlingResult gatling_destroy_pipeline(
  cgpu_device device,
  gatling_pipeline pipeline)
{
  CgpuResult c_result = cgpu_destroy_shader(
    device,
    pipeline.shader
  );
  if (c_result != CGPU_OK) {
    return GATLING_FAIL_UNABLE_TO_DESTROY_SHADER;
  }

  c_result = cgpu_destroy_pipeline(
    device,
    pipeline.pipeline
  );
  if (c_result != CGPU_OK) {
    return GATLING_FAIL_UNABLE_TO_DESTROY_PIPELINE;
  }

  return GATLING_OK;
}

void gatling_get_parent_directory(
  const char* file_path,
  char* dir_path)
{
  char* last_slash = strrchr(file_path, '/');
  char* last_backslash = strrchr(file_path, '\\');
  char* last_path_separator =
      (last_slash > last_backslash) ? last_slash : last_backslash;
  const uint32_t char_index = last_path_separator - file_path;
  if (last_path_separator)
  {
    memccpy(dir_path, file_path, 1, char_index);
    dir_path[char_index] = '\0';
  }
  else
  {
    memccpy(dir_path, file_path, 1, 4096);
  }
}

int main(int argc, const char* argv[])
{
  if (argc != 3) {
    printf("Usage: gatling <scene.gsd> <test.png>\n");
    return 1;
  }

  /* Set up instance and device. */
  CgpuResult c_result = cgpu_initialize("gatling", 0, 1, 0);
  assert(c_result == CGPU_OK);

  uint32_t device_count;
  c_result = cgpu_get_device_count(&device_count);
  assert(c_result == CGPU_OK);
  assert(device_count > 0);

  cgpu_device device;
  c_result = cgpu_create_device(
    0,
    0,
    NULL,
    &device
  );
  assert(c_result == CGPU_OK);

  /* Load scene. */
  uint8_t* scene_data;
  size_t scene_data_size;
  GatlingResult g_result = gatling_read_file(
    argv[1],
    &scene_data,
    &scene_data_size
  );
  assert(g_result == GATLING_OK);

  /* Create input and output buffers. */
  const size_t output_buffer_size_in_floats =
    IMAGE_WIDTH * IMAGE_HEIGHT * 4;
  const size_t output_buffer_size_in_bytes =
    IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float) * 4;

  const size_t input_buffer_size_in_bytes = scene_data_size;

  const size_t path_segment_buffer_size_in_bytes =
    (IMAGE_WIDTH *         /* x dim */
      IMAGE_HEIGHT *       /* y dim */
      NUM_SAMPLES *        /* sample count */
      sizeof(float) * 8) + /* path_segment struct size */
      16;                  /* counter in first 4 bytes + padding */

  cgpu_buffer staging_buffer_in;
  cgpu_buffer input_buffer;
  cgpu_buffer path_segment_buffer;
  cgpu_buffer output_buffer;
  cgpu_buffer staging_buffer_out;

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
    input_buffer_size_in_bytes,
    &staging_buffer_in
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    input_buffer_size_in_bytes,
    &input_buffer
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    path_segment_buffer_size_in_bytes,
    &path_segment_buffer
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    output_buffer_size_in_bytes,
    &output_buffer
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
    output_buffer_size_in_bytes,
    &staging_buffer_out
  );
  assert(c_result == CGPU_OK);

  void* mapped_mem;
  c_result = cgpu_map_buffer(
    device,
    staging_buffer_in,
    0,
    CGPU_WHOLE_SIZE,
    &mapped_mem
  );
  assert(c_result == CGPU_OK);

  memcpy(
    mapped_mem,
    scene_data,
    scene_data_size
  );

  c_result = cgpu_unmap_buffer(
    device,
    staging_buffer_in
  );
  assert(c_result == CGPU_OK);

  cgpu_command_buffer command_buffer;
  c_result = cgpu_create_command_buffer(
    device,
    &command_buffer
  );
  assert(c_result == CGPU_OK);

  /* Set up pipelines. */
  const uint32_t node_offset     = *((uint32_t*) (scene_data +  0));
  const uint32_t node_count      = *((uint32_t*) (scene_data +  4));
  const uint32_t face_offset     = *((uint32_t*) (scene_data +  8));
  const uint32_t face_count      = *((uint32_t*) (scene_data + 12));
  const uint32_t vertex_offset   = *((uint32_t*) (scene_data + 16));
  const uint32_t vertex_count    = *((uint32_t*) (scene_data + 20));
  const uint32_t material_offset = *((uint32_t*) (scene_data + 24));
  const uint32_t material_count  = *((uint32_t*) (scene_data + 28));

  const size_t num_shader_resource_buffers = 7;
  cgpu_shader_resource_buffer shader_resource_buffers[] = {
    { 0u,       output_buffer,              0u,                 CGPU_WHOLE_SIZE },
    { 1u, path_segment_buffer,              0u,                 CGPU_WHOLE_SIZE },
    { 2u,        input_buffer,              0u,                     node_offset },
    { 3u,        input_buffer,     node_offset,       face_offset - node_offset },
    { 4u,        input_buffer,     face_offset,     vertex_offset - face_offset },
    { 5u,        input_buffer,   vertex_offset, material_offset - vertex_offset },
    { 6u,        input_buffer, material_offset,                 CGPU_WHOLE_SIZE }
  };

  char dir_path[4096];
  gatling_get_parent_directory(argv[0], dir_path);

  char prim_ray_gen_shader_path[4096];
  snprintf(prim_ray_gen_shader_path, 4096, "%s/shaders/prim_ray_gen.comp.spv", dir_path);
  char trace_ray_shader_path[4096];
  snprintf(trace_ray_shader_path, 4096, "%s/shaders/trace_ray.comp.spv", dir_path);

  gatling_pipeline pipeline_p1;
  gatling_pipeline pipeline_p2;
  g_result = gatling_create_pipeline(
    device,
    prim_ray_gen_shader_path,
    shader_resource_buffers,
    num_shader_resource_buffers,
    &pipeline_p1
  );
  assert(g_result == GATLING_OK);

  g_result = gatling_create_pipeline(
    device,
    trace_ray_shader_path,
    shader_resource_buffers,
    num_shader_resource_buffers,
    &pipeline_p2
  );
  assert(g_result == GATLING_OK);

  c_result = cgpu_begin_command_buffer(command_buffer);
  assert(c_result == CGPU_OK);

  /* Copy staging buffer to input buffer. */

  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    staging_buffer_in,
    0,
    input_buffer,
    0,
    CGPU_WHOLE_SIZE
  );
  assert(c_result == CGPU_OK);

  cgpu_buffer_memory_barrier buffer_memory_barrier_1 = {};
  buffer_memory_barrier_1.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE;
  buffer_memory_barrier_1.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_1.buffer = input_buffer;
  buffer_memory_barrier_1.byte_offset = 0;
  buffer_memory_barrier_1.byte_count = input_buffer_size_in_bytes;

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &buffer_memory_barrier_1,
    0, NULL
  );
  assert(c_result == CGPU_OK);

  /* Generate primary rays and clear pixels. */

  c_result = cgpu_cmd_bind_pipeline(
    command_buffer,
    pipeline_p1.pipeline
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_cmd_dispatch(
    command_buffer,
    (IMAGE_WIDTH / 32) + 1,
    (IMAGE_HEIGHT / 32) + 1,
    1
  );
  assert(c_result == CGPU_OK);

  /* Trace rays. */

  cgpu_buffer_memory_barrier buffer_memory_barrier_2 = {};
  buffer_memory_barrier_2.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_2.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_2.buffer = path_segment_buffer;
  buffer_memory_barrier_2.byte_offset = 0;
  buffer_memory_barrier_2.byte_count = path_segment_buffer_size_in_bytes;

  cgpu_buffer_memory_barrier buffer_memory_barrier_3 = {};
  buffer_memory_barrier_3.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_3.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_3.buffer = output_buffer;
  buffer_memory_barrier_3.byte_offset = 0;
  buffer_memory_barrier_3.byte_count = output_buffer_size_in_bytes;

  cgpu_buffer_memory_barrier buffer_memory_barrier_2_and_3[] = {
    buffer_memory_barrier_2, buffer_memory_barrier_3
  };

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    2, buffer_memory_barrier_2_and_3,
    0, NULL
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_cmd_bind_pipeline(
    command_buffer,
    pipeline_p2.pipeline
  );
  assert(c_result == CGPU_OK);

  cgpu_physical_device_limits device_limits;
  c_result = cgpu_get_physical_device_limits(
    device,
    &device_limits
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_cmd_dispatch(
    command_buffer,
    /* TODO: what is the optimal number? */
    device_limits.maxComputeWorkGroupInvocations,
    1,
    1
  );
  assert(c_result == CGPU_OK);

  /* Copy staging buffer to output buffer. */

  cgpu_buffer_memory_barrier buffer_memory_barrier_4 = {};
  buffer_memory_barrier_4.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_4.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ;
  buffer_memory_barrier_4.buffer = output_buffer;
  buffer_memory_barrier_4.byte_offset = 0;
  buffer_memory_barrier_4.byte_count = output_buffer_size_in_bytes;

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &buffer_memory_barrier_4,
    0, NULL
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    output_buffer,
    0,
    staging_buffer_out,
    0,
    CGPU_WHOLE_SIZE
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_end_command_buffer(command_buffer);
  assert(c_result == CGPU_OK);

  cgpu_fence fence;
  c_result = cgpu_create_fence(device, &fence);
  assert(c_result == CGPU_OK);

  c_result = cgpu_reset_fence(device, fence);
  assert(c_result == CGPU_OK);

  c_result = cgpu_submit_command_buffer(
    device,
    command_buffer,
    fence
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_wait_for_fence(device, fence);
  assert(c_result == CGPU_OK);

  /* Read data from gpu. */
  float* image_data = malloc(output_buffer_size_in_bytes);

  c_result = cgpu_map_buffer(
    device,
    staging_buffer_out,
    0,
    CGPU_WHOLE_SIZE,
    &mapped_mem
  );
  assert(c_result == CGPU_OK);

  memcpy(
    image_data,
    mapped_mem,
    output_buffer_size_in_bytes
  );

  c_result = cgpu_unmap_buffer(
    device,
    staging_buffer_out
  );
  assert(c_result == CGPU_OK);

  /* Save image. */
  g_result = gatling_save_img(
    image_data,
    output_buffer_size_in_floats,
    argv[2]
  );
  assert(g_result == GATLING_OK);

  /* Clean up. */
  free(image_data);

  c_result = cgpu_destroy_fence(
    device,
    fence
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_destroy_command_buffer(
    device,
    command_buffer
  );
  assert(c_result == CGPU_OK);

  gatling_destroy_pipeline(
    device,
    pipeline_p1
  );
  gatling_destroy_pipeline(
    device,
    pipeline_p2
  );

  c_result = cgpu_destroy_buffer(device, staging_buffer_in);
  assert(c_result == CGPU_OK);
  c_result = cgpu_destroy_buffer(device, input_buffer);
  assert(c_result == CGPU_OK);
  c_result = cgpu_destroy_buffer(device, path_segment_buffer);
  assert(c_result == CGPU_OK);
  c_result = cgpu_destroy_buffer(device, output_buffer);
  assert(c_result == CGPU_OK);
  c_result = cgpu_destroy_buffer(device, staging_buffer_out);
  assert(c_result == CGPU_OK);

  c_result = cgpu_destroy_device(device);
  assert(c_result == CGPU_OK);

  c_result = cgpu_destroy();
  assert(c_result == CGPU_OK);

  return 0;
}
