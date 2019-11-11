#include <cgpu/cgpu.hpp>

#include <cassert>
#include <cstdint>
#include <vector>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef enum GatlingResult {
  GATLING_OK = 0,
  GATLING_FAIL_UNABLE_TO_WRITE_OUTPUT_IMAGE = -1,
  GATLING_FAIL_UNABLE_TO_OPEN_FILE = -2,
  GATLING_FAIL_UNABLE_TO_CREATE_SHADER = -3,
  GATLING_FAIL_UNABLE_TO_CREATE_PIPELINE = -4,
  GATLING_FAIL_UNABLE_TO_DESTROY_SHADER = -5,
  GATLING_FAIL_UNABLE_TO_DESTROY_PIPELINE = -6
} GatlingResult;

GatlingResult gatling_save_img(
  std::vector<float>& data,
  const char* file_path)
{
  std::vector<uint8_t> rgba_bytes;
  rgba_bytes.resize(data.size());

  size_t i = 0;
  for (const float channel : data) {
    rgba_bytes[i] = (uint8_t)(channel * 255.0f);
    i++;
  }

  stbi_flip_vertically_on_write(true);

  const int result = stbi_write_png(
    file_path,
    1024,
    1024,
    4,
    rgba_bytes.data(),
    1024 * 4
  );

  if (!result) {
    return GATLING_FAIL_UNABLE_TO_WRITE_OUTPUT_IMAGE;
  }

  return GATLING_OK;
}

GatlingResult gatling_read_file(
  std::vector<std::uint8_t>& data,
  const char* file_path)
{
  std::ifstream file{
    file_path,
    std::ios_base::in | std::ios_base::binary
  };
  if (!file.is_open()) {
    return GATLING_FAIL_UNABLE_TO_OPEN_FILE;
  }
  file.seekg(0, std::ios_base::end);
  data.resize(file.tellg());
  file.seekg(0, std::ios_base::beg);
  file.read(
    reinterpret_cast<char*>(data.data()),
    data.size()
  );
  return GATLING_OK;
}

struct gatling_compute_pipeline
{
  cgpu_pipeline pipeline;
  cgpu_shader shader;
};

GatlingResult gatling_create_compute_pipeline(
  const cgpu_device& device,
  const char* shader_file_path,
  const std::vector<cgpu_shader_resource_buffer>& shader_resource_buffers,
  gatling_compute_pipeline& pipeline)
{
  std::vector<std::uint8_t> shader_source;

  GatlingResult g_result = gatling_read_file(
    shader_source,
    shader_file_path
  );
  if (g_result != GATLING_OK) {
    return g_result;
  }

  CgpuResult c_result = cgpu_create_shader(
    device,
    shader_source.size(),
    shader_source.data(),
    pipeline.shader
  );
  if (c_result != CGPU_OK) {
    return GATLING_FAIL_UNABLE_TO_CREATE_PIPELINE;
  }

  c_result = cgpu_create_pipeline(
    device,
    shader_resource_buffers.size(),
    shader_resource_buffers.data(),
    0,
    nullptr,
    pipeline.shader,
    "main",
    pipeline.pipeline
  );
  if (c_result != CGPU_OK) {
    return GATLING_FAIL_UNABLE_TO_CREATE_PIPELINE;
  }

  return GATLING_OK;
}

GatlingResult gatling_destroy_compute_pipeline(
  const cgpu_device& device,
  const gatling_compute_pipeline& pipeline)
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
    nullptr,
    device
  );
  assert(c_result == CGPU_OK);

  // Load scene.
  std::vector<std::uint8_t> scene_data;
  GatlingResult g_result = gatling_read_file(
    scene_data,
    "/Users/pablode/Dropbox/Projects/gatling/build/test.gsd"
  );
  assert(g_result == GATLING_OK);

  // Create input and output buffers.
  const uint32_t image_width = 1024;
  const uint32_t image_height = 1024;

  const uint32_t output_buffer_size_in_floats =
    image_width * image_height * 4;
  const uint32_t output_buffer_size_in_bytes =
    image_width * image_height * sizeof(float) * 4;

  const uint32_t input_buffer_size_in_bytes =
    scene_data.size();

  const uint32_t path_segment_buffer_size_in_bytes =
    (image_width *       // x dim
      image_height *     // y dim
      4 *                // sample count
      sizeof(float) * 8) // path_segment struct size
      + 16;              // counter in first 4 bytes + padding

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
    staging_buffer_in
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    input_buffer_size_in_bytes,
    input_buffer
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    path_segment_buffer_size_in_bytes,
    path_segment_buffer
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    output_buffer_size_in_bytes,
    output_buffer
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
    output_buffer_size_in_bytes,
    staging_buffer_out
  );
  assert(c_result == CGPU_OK);

  void* mapped_mem;
  c_result = cgpu_map_buffer(
    device,
    staging_buffer_in,
    &mapped_mem
  );
  assert(c_result == CGPU_OK);

  std::memcpy(
    mapped_mem,
    scene_data.data(),
    scene_data.size()
  );

  c_result = cgpu_unmap_buffer(
    device,
    staging_buffer_in
  );
  assert(c_result == CGPU_OK);

  cgpu_command_buffer command_buffer;
  c_result = cgpu_create_command_buffer(
    device,
    command_buffer
  );
  assert(c_result == CGPU_OK);

  // Execute pipelines.
  std::vector<cgpu_shader_resource_buffer> shader_resource_buffers = {
    { 0, CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE, output_buffer       },
    { 1, CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE, input_buffer        },
    { 2, CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE, input_buffer        },
    { 3, CGPU_SHADER_RESOURCE_USAGE_FLAG_WRITE, path_segment_buffer }
  };

  gatling_compute_pipeline pipeline_p1;
  gatling_compute_pipeline pipeline_p2;

  g_result = gatling_create_compute_pipeline(
    device,
    "/Users/pablode/Dropbox/Projects/gatling/build/bin/gatling/shaders/prim_ray_gen.comp.spv",
    shader_resource_buffers,
    pipeline_p1
  );
  assert(g_result == GATLING_OK);

  g_result = gatling_create_compute_pipeline(
    device,
    "/Users/pablode/Dropbox/Projects/gatling/build/bin/gatling/shaders/trace_ray.comp.spv",
    shader_resource_buffers,
    pipeline_p2
  );
  assert(g_result == GATLING_OK);

  c_result = cgpu_begin_command_buffer(command_buffer);
  assert(c_result == CGPU_OK);

  // Copy staging buffer to input buffer.

  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    staging_buffer_in,
    input_buffer
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
    0, nullptr,
    1, &buffer_memory_barrier_1,
    0, nullptr
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
    0, nullptr,
    2, buffer_memory_barrier_2_and_3,
    0, nullptr
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
    device_limits
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
    0, nullptr,
    1, &buffer_memory_barrier_4,
    0, nullptr
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    output_buffer,
    staging_buffer_out
  );
  assert(c_result == CGPU_OK);

  c_result = cgpu_end_command_buffer(command_buffer);
  assert(c_result == CGPU_OK);

  cgpu_fence fence;
  c_result = cgpu_create_fence(device, fence);
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
  std::vector<float> image_data;
  image_data.resize(output_buffer_size_in_floats);

  c_result = cgpu_map_buffer(
    device,
    staging_buffer_out,
    &mapped_mem
  );
  assert(c_result == CGPU_OK);

  std::memcpy(
    image_data.data(),
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
    "test.png"
  );
  assert(g_result == GATLING_OK);

  // Clean up.
  c_result = cgpu_destroy_fence(
    device,
    fence
  );

  c_result = cgpu_destroy_command_buffer(
    device,
    command_buffer
  );
  assert(c_result == CGPU_OK);

  gatling_destroy_compute_pipeline(
    device,
    pipeline_p1
  );
  gatling_destroy_compute_pipeline(
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
