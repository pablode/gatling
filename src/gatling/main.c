#include <cgpu/cgpu.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
    const float color = data[i];
    const uint8_t color_quant = (uint8_t) (color * 255.0f);
    temp_data[i] = color_quant;
  }

  stbi_flip_vertically_on_write(true);

  const int result = stbi_write_png(
    file_path,
    1024,
    1024,
    4,
    temp_data,
    1024 * 4
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
    data_size, // TODO: swap args?
    data,
    &pipeline->shader
  );
  if (c_result != CGPU_OK) {
    return GATLING_FAIL_UNABLE_TO_CREATE_PIPELINE;
  }

  free(data);

  c_result = cgpu_create_pipeline(
    device,
    num_shader_resource_buffers, // TOOD: swap args?
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

int main(int argc, const char* argv[])
{
  if (argc != 2) {
    printf("Please enter the gatling scene description (gsd) file path.\n");
    return 1;
  }

  // Set up instance and device.
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

  // Load scene.
  uint8_t* scene_data;
  size_t scene_data_size;
  GatlingResult g_result = gatling_read_file(
    argv[1],
    &scene_data,
    &scene_data_size
  );
  assert(g_result == GATLING_OK);

  // Create input and output buffers.
  const size_t image_width = 1024;
  const size_t image_height = 1024;

  const size_t output_buffer_size_in_floats =
    image_width * image_height * 4;
  const size_t output_buffer_size_in_bytes =
    image_width * image_height * sizeof(float) * 4;

  const size_t input_buffer_size_in_bytes = scene_data_size;

  const size_t path_segment_buffer_size_in_bytes =
    (image_width *         // x dim
      image_height *       // y dim
      4 *                  // sample count
      sizeof(float) * 8) + // path_segment struct size
      16;                  // counter in first 4 bytes + padding

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

  // Execute pipelines.
  const size_t num_shader_resource_buffers = 4;
  cgpu_shader_resource_buffer shader_resource_buffers[] = {
    { 0, CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE, output_buffer       },
    { 1, CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE, input_buffer        }, // TODO
    { 2, CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE, input_buffer        },
    { 3, CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE, path_segment_buffer }
  };

  gatling_pipeline pipeline_p1;
  gatling_pipeline pipeline_p2;
  g_result = gatling_create_pipeline(
    device,
    "/home/pablode/Dropbox/projects/gatling/build/bin/gatling/shaders/prim_ray_gen.comp.spv",
    shader_resource_buffers,
    num_shader_resource_buffers,
    &pipeline_p1
  );
  assert(g_result == GATLING_OK);

  g_result = gatling_create_pipeline(
    device,
    "/home/pablode/Dropbox/projects/gatling/build/bin/gatling/shaders/trace_ray.comp.spv",
    shader_resource_buffers,
    num_shader_resource_buffers,
    &pipeline_p2
  );
  assert(g_result == GATLING_OK);

  c_result = cgpu_begin_command_buffer(command_buffer);
  assert(c_result == CGPU_OK);

  // Copy staging buffer to input buffer.

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
  buffer_memory_barrier_1.num_bytes = input_buffer_size_in_bytes;

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &buffer_memory_barrier_1,
    0, NULL
  );
  assert(c_result == CGPU_OK);

  // Generate primary rays and clear pixels.

  c_result = cgpu_cmd_bind_pipeline(
    command_buffer,
    pipeline_p1.pipeline
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_cmd_dispatch(
    command_buffer,
    1024 / 32,
    1024 / 32,
    1
  );
  assert(c_result == CGPU_OK);

  // Trace rays.

  cgpu_buffer_memory_barrier buffer_memory_barrier_2 = {};
  buffer_memory_barrier_2.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_2.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_2.buffer = path_segment_buffer;
  buffer_memory_barrier_2.byte_offset = 0;
  buffer_memory_barrier_2.num_bytes = path_segment_buffer_size_in_bytes;

  cgpu_buffer_memory_barrier buffer_memory_barrier_3 = {};
  buffer_memory_barrier_3.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_3.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_3.buffer = output_buffer;
  buffer_memory_barrier_3.byte_offset = 0;
  buffer_memory_barrier_3.num_bytes = output_buffer_size_in_bytes;

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
    // TODO: what is the optimal number?
    device_limits.maxComputeWorkGroupInvocations,
    1,
    1
  );
  assert(c_result == CGPU_OK);

  // Copy staging buffer to output buffer.

  cgpu_buffer_memory_barrier buffer_memory_barrier_4 = {};
  buffer_memory_barrier_4.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_4.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ;
  buffer_memory_barrier_4.buffer = output_buffer;
  buffer_memory_barrier_4.byte_offset = 0;
  buffer_memory_barrier_4.num_bytes = output_buffer_size_in_bytes;

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

  // Read data from gpu.
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

  // Save image.
  g_result = gatling_save_img(
    image_data,
    output_buffer_size_in_floats,
    "test.png"
  );
  assert(g_result == GATLING_OK);

  // Clean up.
  free(image_data);

  c_result = cgpu_destroy_fence(
    device,
    fence
  );

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
