#include <stdint.h>
#include <stdio.h>

#include <cgpu.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_WIDTH 3840
#define IMAGE_HEIGHT 2160
#define NUM_SAMPLES 1

#define gatling_fail(msg)                                                         \
  do {                                                                            \
    printf("Gatling encountered a fatal error at line %d: %s\n", __LINE__, msg);  \
    exit(EXIT_FAILURE);                                                           \
  } while(0)

static void gatling_cgpu_ensure(CgpuResult result)
{
  if (result != CGPU_OK) {
    printf("Gatling encountered a fatal error.");
    exit(EXIT_FAILURE);
  }
}

static void gatling_cgpu_warn(CgpuResult result, const char *msg)
{
  if (result != CGPU_OK) {
    printf("Gatling encountered an error: %s\n", msg);
  }
}

static void gatling_save_img(
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
    gatling_fail("Unable to save image.");
  }
}

static void gatling_read_file(
  const char* file_path,
  uint8_t** data,
  size_t* data_size)
{
  FILE *file = fopen(file_path, "rb");
  if (file == NULL) {
    gatling_fail("Unable to open file for reading.");
  }

  fseeko(file, 0, SEEK_END);
  *data_size = ftello(file);
  fseeko(file, 0, SEEK_SET);

  *data = malloc(*data_size);
  fread(*data, 1, *data_size, file);

  const int close_result = fclose(file);
  if (close_result != 0) {
    printf("Unable to close file '%s'.", file_path);
  }
}

typedef struct gatling_pipeline {
  cgpu_pipeline pipeline;
  cgpu_shader shader;
} gatling_pipeline;

static void gatling_create_pipeline(
  cgpu_device device,
  const char *shader_file_path,
  cgpu_shader_resource_buffer *shader_resource_buffers,
  size_t num_shader_resource_buffers,
  gatling_pipeline *pipeline)
{
  uint32_t* data;
  size_t data_size;

  gatling_read_file(
    shader_file_path,
    (uint8_t**) &data,
    &data_size
  );

  CgpuResult c_result = cgpu_create_shader(
    device,
    data_size,
    data,
    &pipeline->shader
  );
  if (c_result != CGPU_OK) {
    gatling_fail("Unable to create shader.");
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
  gatling_cgpu_ensure(c_result);
}

static void gatling_destroy_pipeline(
  cgpu_device device,
  gatling_pipeline pipeline)
{
  CgpuResult c_result = cgpu_destroy_shader(
    device,
    pipeline.shader
  );
  gatling_cgpu_warn(c_result, "Unable to destroy shader.");

  c_result = cgpu_destroy_pipeline(
    device,
    pipeline.pipeline
  );
  gatling_cgpu_warn(c_result, "Unable to destroy pipeline.");
}

static void gatling_get_parent_directory(
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

static void gatling_print_timestamp(
  const char* name,
  uint64_t elapsed_timesteps,
  float timestamp_period)
{
  const float elapsed_nanoseconds  = elapsed_timesteps * timestamp_period;
  const float elapsed_microseconds = elapsed_nanoseconds / 1000.0f;
  const float elapsed_milliseconds = elapsed_microseconds / 1000.0f;
  printf("Elapsed time for %s: %.2fms\n", name, elapsed_milliseconds);
}

int main(int argc, const char* argv[])
{
  if (argc != 3) {
    printf("Usage: gatling <scene.gsd> <test.png>\n");
    return 1;
  }

  /* Set up instance and device. */
  CgpuResult c_result = cgpu_initialize(
    "gatling",
    GATLING_VERSION_MAJOR,
    GATLING_VERSION_MINOR,
    GATLING_VERSION_PATCH
  );
  if (c_result != CGPU_OK) {
    gatling_fail("Unable to initialize cgpu.");
  }

  uint32_t device_count;
  c_result = cgpu_get_device_count(&device_count);
  if (c_result != CGPU_OK || device_count == 0) {
    gatling_fail("Unable to find device.");
  }

  cgpu_device device;
  c_result = cgpu_create_device(
    0,
    0,
    NULL,
    &device
  );
  if (c_result != CGPU_OK) {
    gatling_fail("Unable to create device.");
  }

  cgpu_physical_device_limits device_limits;
  c_result = cgpu_get_physical_device_limits(
    device,
    &device_limits
  );
  gatling_cgpu_ensure(c_result);

  /* Load scene. */
  uint8_t* scene_data;
  size_t scene_data_size;
  gatling_read_file(
    argv[1],
    &scene_data,
    &scene_data_size
  );

  /* Create input and output buffers. */
  const size_t output_buffer_size_in_floats =
    IMAGE_WIDTH * IMAGE_HEIGHT * 4;
  const size_t output_buffer_size_in_bytes =
    IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float) * 4;

  const size_t input_buffer_size_in_bytes = scene_data_size;

  const size_t path_segment_buffer_size_in_bytes =
    IMAGE_WIDTH *   /* x dim */
    IMAGE_HEIGHT *  /* y dim */
    NUM_SAMPLES *   /* sample count */
    32 +            /* path_segment struct byte size */
    16;             /* counter in first 4 bytes + padding */

  const size_t hit_info_buffer_size_in_bytes =
    IMAGE_WIDTH *   /* x dim */
    IMAGE_HEIGHT *  /* y dim */
    NUM_SAMPLES *   /* sample count */
    32 +            /* hit_size struct byte size */
    16;             /* counter in first 4 bytes + padding */

  cgpu_buffer staging_buffer_in;
  cgpu_buffer input_buffer;
  cgpu_buffer path_segment_buffer;
  cgpu_buffer hit_info_buffer;
  cgpu_buffer output_buffer;
  cgpu_buffer staging_buffer_out;
  cgpu_buffer timestamp_buffer;

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
    input_buffer_size_in_bytes,
    &staging_buffer_in
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    input_buffer_size_in_bytes,
    &input_buffer
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    path_segment_buffer_size_in_bytes,
    &path_segment_buffer
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    hit_info_buffer_size_in_bytes,
    &hit_info_buffer
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    output_buffer_size_in_bytes,
    &output_buffer
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
    output_buffer_size_in_bytes,
    &staging_buffer_out
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT,
    32 * sizeof(uint64_t),
    &timestamp_buffer
  );
  gatling_cgpu_ensure(c_result);

  void* mapped_mem;
  c_result = cgpu_map_buffer(
    device,
    staging_buffer_in,
    0,
    CGPU_WHOLE_SIZE,
    &mapped_mem
  );
  gatling_cgpu_ensure(c_result);

  memcpy(
    mapped_mem,
    scene_data,
    scene_data_size
  );

  c_result = cgpu_unmap_buffer(
    device,
    staging_buffer_in
  );
  gatling_cgpu_ensure(c_result);

  cgpu_command_buffer command_buffer;
  c_result = cgpu_create_command_buffer(
    device,
    &command_buffer
  );
  gatling_cgpu_ensure(c_result);

  /* Set up pipelines. */
  const uint32_t node_offset     = *((uint32_t*) (scene_data +  0));
  const uint32_t face_offset     = *((uint32_t*) (scene_data +  8));
  const uint32_t vertex_offset   = *((uint32_t*) (scene_data + 16));
  const uint32_t material_offset = *((uint32_t*) (scene_data + 24));

  const size_t num_shader_resource_buffers = 8;
  cgpu_shader_resource_buffer shader_resource_buffers[] = {
    { 0u,       output_buffer,              0u,                 CGPU_WHOLE_SIZE },
    { 1u, path_segment_buffer,              0u,                 CGPU_WHOLE_SIZE },
    { 2u,        input_buffer,              0u,                     node_offset },
    { 3u,        input_buffer,     node_offset,       face_offset - node_offset },
    { 4u,        input_buffer,     face_offset,     vertex_offset - face_offset },
    { 5u,        input_buffer,   vertex_offset, material_offset - vertex_offset },
    { 6u,        input_buffer, material_offset,                 CGPU_WHOLE_SIZE },
    { 7u,    hit_info_buffer,               0u,                 CGPU_WHOLE_SIZE }
  };

  char dir_path[4096];
  gatling_get_parent_directory(argv[0], dir_path);

  char kernel_ray_gen_shader_path[4096];
  snprintf(kernel_ray_gen_shader_path, 4096, "%s/shaders/kernel_ray_gen.comp.spv", dir_path);
  char kernel_extend_shader_path[4096];
  snprintf(kernel_extend_shader_path, 4096, "%s/shaders/kernel_extend.comp.spv", dir_path);
  char kernel_shade_shader_path[4096];
  snprintf(kernel_shade_shader_path, 4096, "%s/shaders/kernel_shade.comp.spv", dir_path);

  gatling_pipeline pipeline_ray_gen;
  gatling_pipeline pipeline_extend;
  gatling_pipeline pipeline_shade;

  gatling_create_pipeline(
    device,
    kernel_ray_gen_shader_path,
    shader_resource_buffers,
    num_shader_resource_buffers,
    &pipeline_ray_gen
  );
  gatling_create_pipeline(
    device,
    kernel_extend_shader_path,
    shader_resource_buffers,
    num_shader_resource_buffers,
    &pipeline_extend
  );
  gatling_create_pipeline(
    device,
    kernel_shade_shader_path,
    shader_resource_buffers,
    num_shader_resource_buffers,
    &pipeline_shade
  );

  c_result = cgpu_begin_command_buffer(command_buffer);
  gatling_cgpu_ensure(c_result);

  /* Write start timestamp. */
  c_result = cgpu_cmd_reset_timestamps(
    command_buffer,
    0u,
    32u
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_write_timestamp(command_buffer, 0u);
  gatling_cgpu_ensure(c_result);

  /* Copy staging buffer to input buffer. */
  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    staging_buffer_in,
    0,
    input_buffer,
    0,
    CGPU_WHOLE_SIZE
  );
  gatling_cgpu_ensure(c_result);

  cgpu_buffer_memory_barrier buffer_memory_barrier_1;
  buffer_memory_barrier_1.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_WRITE;
  buffer_memory_barrier_1.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_1.buffer = input_buffer;
  buffer_memory_barrier_1.byte_offset = 0;
  buffer_memory_barrier_1.byte_count = CGPU_WHOLE_SIZE;

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &buffer_memory_barrier_1,
    0, NULL
  );
  gatling_cgpu_ensure(c_result);

  /* Generate primary rays and clear pixels. */
  c_result = cgpu_cmd_bind_pipeline(
    command_buffer,
    pipeline_ray_gen.pipeline
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_write_timestamp(command_buffer, 1u);
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_dispatch(
    command_buffer,
    (IMAGE_WIDTH / 32) + 1,
    (IMAGE_HEIGHT / 32) + 1,
    1
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_write_timestamp(command_buffer, 2u);
  gatling_cgpu_ensure(c_result);

  /* Trace rays. */
  cgpu_buffer_memory_barrier buffer_memory_barrier_2;
  buffer_memory_barrier_2.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_2.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_2.buffer = path_segment_buffer;
  buffer_memory_barrier_2.byte_offset = 0u;
  buffer_memory_barrier_2.byte_count = CGPU_WHOLE_SIZE;

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &buffer_memory_barrier_2,
    0, NULL
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_bind_pipeline(
    command_buffer,
    pipeline_extend.pipeline
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_write_timestamp(command_buffer, 3u);
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_dispatch(
    command_buffer,
    /* TODO: what is the optimal number? */
    device_limits.maxComputeWorkGroupInvocations,
    1,
    1
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_write_timestamp(command_buffer, 4u);
  gatling_cgpu_ensure(c_result);

  /* Shade hit points. */
  cgpu_buffer_memory_barrier buffer_memory_barrier_3;
  buffer_memory_barrier_3.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_3.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_3.buffer = hit_info_buffer;
  buffer_memory_barrier_3.byte_offset = 0u;
  buffer_memory_barrier_3.byte_count = CGPU_WHOLE_SIZE;

  cgpu_buffer_memory_barrier buffer_memory_barrier_4;
  buffer_memory_barrier_4.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_4.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_4.buffer = output_buffer;
  buffer_memory_barrier_4.byte_offset = 0u;
  buffer_memory_barrier_4.byte_count = CGPU_WHOLE_SIZE;

  cgpu_buffer_memory_barrier buffer_memory_barrier_3_and_4[] = {
    buffer_memory_barrier_3, buffer_memory_barrier_4
  };

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    2, buffer_memory_barrier_3_and_4,
    0, NULL
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_bind_pipeline(
    command_buffer,
    pipeline_shade.pipeline
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_write_timestamp(command_buffer, 5u);
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_dispatch(
    command_buffer,
    /* TODO: what is the optimal number? */
    device_limits.maxComputeWorkGroupInvocations,
    1,
    1
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_write_timestamp(command_buffer, 6u);
  gatling_cgpu_ensure(c_result);

  /* Copy staging buffer to output buffer. */
  cgpu_buffer_memory_barrier buffer_memory_barrier_5;
  buffer_memory_barrier_5.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_5.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_TRANSFER_READ;
  buffer_memory_barrier_5.buffer = output_buffer;
  buffer_memory_barrier_5.byte_offset = 0;
  buffer_memory_barrier_5.byte_count = CGPU_WHOLE_SIZE;

  c_result = cgpu_cmd_pipeline_barrier(
    command_buffer,
    0, NULL,
    1, &buffer_memory_barrier_5,
    0, NULL
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_copy_buffer(
    command_buffer,
    output_buffer,
    0,
    staging_buffer_out,
    0,
    CGPU_WHOLE_SIZE
  );
  gatling_cgpu_ensure(c_result);

  /* Write end timestamp and copy timestamps. */
  c_result = cgpu_cmd_write_timestamp(command_buffer, 7u);
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_copy_timestamps(
    command_buffer,
    timestamp_buffer,
    0u,
    8u,
    true
  );
  gatling_cgpu_ensure(c_result);

  /* End and submit command buffer. */
  c_result = cgpu_end_command_buffer(command_buffer);
  gatling_cgpu_ensure(c_result);

  cgpu_fence fence;
  c_result = cgpu_create_fence(device, &fence);
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_reset_fence(device, fence);
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_submit_command_buffer(
    device,
    command_buffer,
    fence
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_wait_for_fence(device, fence);
  gatling_cgpu_ensure(c_result);

  /* Read timestamps. */
  uint64_t* timestamps;

  c_result = cgpu_map_buffer(
    device,
    timestamp_buffer,
    0,
    CGPU_WHOLE_SIZE,
    (void**) &timestamps
  );
  gatling_cgpu_ensure(c_result);

  const uint64_t timestamp_start = timestamps[0];
  const uint64_t timestamp_start_ray_gen = timestamps[1];
  const uint64_t timestamp_end_ray_gen = timestamps[2];
  const uint64_t timestamp_start_extend = timestamps[3];
  const uint64_t timestamp_end_extend = timestamps[4];
  const uint64_t timestamp_start_shade = timestamps[5];
  const uint64_t timestamp_end_shade = timestamps[6];
  const uint64_t timestamp_end = timestamps[7];

  c_result = cgpu_unmap_buffer(device, timestamp_buffer);
  gatling_cgpu_ensure(c_result);

  const float timestamp_ns_period = device_limits.timestampPeriod;
  const uint64_t timespan_ray_gen = timestamp_end_ray_gen - timestamp_start_ray_gen;
  const uint64_t timespan_extend = timestamp_end_extend - timestamp_start_extend;
  const uint64_t timespan_shade = timestamp_end_shade - timestamp_start_shade;
  const uint64_t timespan_total = timestamp_end - timestamp_start;
  const uint64_t timespan_sync = timespan_total - timespan_ray_gen - timespan_extend - timespan_shade;

  gatling_print_timestamp("prim ray gen",  timespan_ray_gen, timestamp_ns_period);
  gatling_print_timestamp("extend",        timespan_extend,  timestamp_ns_period);
  gatling_print_timestamp("shade",         timespan_shade,   timestamp_ns_period);
  gatling_print_timestamp("sync overhead", timespan_sync,    timestamp_ns_period);
  gatling_print_timestamp("total",         timespan_total,   timestamp_ns_period);

  /* Read data from gpu. */
  float* image_data = malloc(output_buffer_size_in_bytes);

  c_result = cgpu_map_buffer(
    device,
    staging_buffer_out,
    0,
    CGPU_WHOLE_SIZE,
    &mapped_mem
  );
  gatling_cgpu_ensure(c_result);

  memcpy(
    image_data,
    mapped_mem,
    output_buffer_size_in_bytes
  );

  c_result = cgpu_unmap_buffer(
    device,
    staging_buffer_out
  );
  gatling_cgpu_ensure(c_result);

  /* Save image. */
  gatling_save_img(
    image_data,
    output_buffer_size_in_floats,
    argv[2]
  );

  /* Clean up. */
  free(image_data);

  c_result = cgpu_destroy_fence(
    device,
    fence
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_destroy_command_buffer(
    device,
    command_buffer
  );
  gatling_cgpu_ensure(c_result);

  gatling_destroy_pipeline(device, pipeline_ray_gen);
  gatling_destroy_pipeline(device, pipeline_extend);
  gatling_destroy_pipeline(device, pipeline_shade);

  c_result = cgpu_destroy_buffer(device, timestamp_buffer);
  gatling_cgpu_ensure(c_result);
  c_result = cgpu_destroy_buffer(device, staging_buffer_in);
  gatling_cgpu_ensure(c_result);
  c_result = cgpu_destroy_buffer(device, input_buffer);
  gatling_cgpu_ensure(c_result);
  c_result = cgpu_destroy_buffer(device, hit_info_buffer);
  gatling_cgpu_ensure(c_result);
  c_result = cgpu_destroy_buffer(device, path_segment_buffer);
  gatling_cgpu_ensure(c_result);
  c_result = cgpu_destroy_buffer(device, output_buffer);
  gatling_cgpu_ensure(c_result);
  c_result = cgpu_destroy_buffer(device, staging_buffer_out);
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_destroy_device(device);
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_destroy();
  gatling_cgpu_ensure(c_result);

  return EXIT_SUCCESS;
}
