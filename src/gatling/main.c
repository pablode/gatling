#include <stdint.h>
#include <stdio.h>

#include <cgpu.h>

#include "mmap.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static uint32_t DEFAULT_IMAGE_WIDTH = 1920;
static uint32_t DEFAULT_IMAGE_HEIGHT = 1080;
static uint32_t DEFAULT_SPP = 4;

#define gatling_fail(msg)                                                         \
  do {                                                                            \
    printf("Gatling encountered a fatal error at line %d: %s\n", __LINE__, msg);  \
    exit(EXIT_FAILURE);                                                           \
  } while(0)

#define gatling_cgpu_ensure(result)                                                         \
  do {                                                                                      \
    if (result != CGPU_OK) {                                                                \
      printf("Gatling encountered a fatal CGPU error at line %d: %d\n", __LINE__, result);  \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

static void gatling_cgpu_warn(CgpuResult result, const char *msg)
{
  if (result != CGPU_OK) {
    printf("Gatling encountered an error: %s\n", msg);
  }
}

static void gatling_save_img_wfunc(void *context, void *data, int size)
{
  gatling_file* file;
  const char* file_path = (const char*) context;
  const bool success = gatling_file_create(file_path, size, &file);
  if (!success) {
    gatling_fail("Unable to open output file.");
  }

  void* mapped_mem = gatling_mmap(file, 0, size);
  if (!mapped_mem) {
    gatling_fail("Unable to map output file.");
  }
  memcpy(mapped_mem, data, size);

  gatling_munmap(file, mapped_mem);
  gatling_file_close(file);
}

static void gatling_save_img(
  const float* data,
  size_t float_count,
  uint32_t image_width,
  uint32_t image_height,
  const char* file_path)
{
  uint8_t* temp_data = malloc(float_count);

  for (size_t i = 0; i < float_count; ++i)
  {
    int32_t color = (int32_t) (data[i] * 255.0f);
    if (color < 0)   { color = 0;   }
    if (color > 255) { color = 255; }
    temp_data[i] = (uint8_t) color;
  }

  stbi_flip_vertically_on_write(true);

  const uint32_t num_components = 4;

  const int result = stbi_write_png_to_func(
    gatling_save_img_wfunc,
    (void*)file_path,
    image_width,
    image_height,
    num_components,
    temp_data,
    image_width * num_components
  );

  free(temp_data);

  if (!result) {
    gatling_fail("Unable to save image.");
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
  uint32_t spec_const_count,
  const cgpu_specialization_constant* spec_constants,
  gatling_pipeline *pipeline)
{
  gatling_file* file;
  uint64_t file_size;

  const bool ok = gatling_file_open(shader_file_path, GATLING_FILE_USAGE_READ, &file);
  if (!ok) {
    gatling_fail("Unable to open shader file.");
  }

  file_size = gatling_file_size(file);

  uint32_t* data = (uint32_t*) gatling_mmap(file, 0, file_size);

  if (!data) {
    gatling_fail("Unable to map shader file.");
  }

  CgpuResult c_result = cgpu_create_shader(
    device,
    file_size,
    data,
    &pipeline->shader
  );

  gatling_munmap(file, data);
  gatling_file_close(file);

  if (c_result != CGPU_OK) {
    gatling_fail("Unable to create shader.");
  }

  c_result = cgpu_create_pipeline(
    device,
    num_shader_resource_buffers,
    shader_resource_buffers,
    0,
    NULL,
    pipeline->shader,
    "main",
    spec_const_count,
    spec_constants,
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

typedef struct program_options {
  const char* input_file;
  const char* output_file;
  uint32_t image_width;
  uint32_t image_height;
  uint32_t spp;
} program_options;

void gatling_print_usage_and_exit()
{
  printf("Usage: gatling <scene.gsd> <test.png> [options]\n");
  printf("\n");
  printf("Options:\n");
  printf("--image-width  [default: %u]\n", DEFAULT_IMAGE_WIDTH);
  printf("--image-height [default: %u]\n", DEFAULT_IMAGE_HEIGHT);
  printf("--spp          [default: %u]\n", DEFAULT_SPP);
  exit(EXIT_FAILURE);
}

void gatling_parse_args(int argc, const char* argv[], program_options* options)
{
  if (argc < 3) {
    gatling_print_usage_and_exit();
  }

  if (strncmp("-", argv[1], 1) == 0 ||
      strncmp("-", argv[2], 1) == 0)
  {
    gatling_print_usage_and_exit();
  }

  options->input_file = argv[1];
  options->output_file = argv[2];
  options->image_width = DEFAULT_IMAGE_WIDTH;
  options->image_height = DEFAULT_IMAGE_HEIGHT;
  options->spp = DEFAULT_SPP;

  for (uint32_t i = 3; i < argc; ++i)
  {
    const char* arg = argv[i];

    if (strncmp("--", arg, 2) != 0) {
      gatling_print_usage_and_exit();
    }

    char* key_value = strdup(&arg[2]);
    char* value = key_value;
    char* key = strsep(&value, "=");

    bool fail = false;

    if (value == NULL) {
      fail = true;
    }
    else if (strcmp(key, "image-width") == 0) {
      char* endptr;
      options->image_width = strtol(value, &endptr, 10);
      fail |= (value == endptr);
    }
    else if (strcmp(key, "image-height") == 0) {
      char* endptr;
      options->image_height = strtol(value, &endptr, 10);
      fail |= (value == endptr);
    }
    else if (strcmp(key, "spp") == 0) {
      char* endptr;
      options->spp = strtol(value, &endptr, 10);
      fail |= (value == endptr);
    }
    else {
      fail = true;
    }

    free(key_value);
    if (fail) {
      gatling_print_usage_and_exit();
    }
  }
}

uint64_t gatling_align_buffer(
  uint64_t offset_alignment,
  uint64_t buffer_size,
  uint64_t* total_size)
{
  const uint64_t offset =
    ((*total_size) + offset_alignment - 1) / offset_alignment
    * offset_alignment;

  (*total_size) = offset + buffer_size;

  return offset;
}

int main(int argc, const char* argv[])
{
  program_options options;
  gatling_parse_args(argc, argv, &options);

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
  c_result = cgpu_create_device(0, &device);
  if (c_result != CGPU_OK) {
    gatling_fail("Unable to create device.");
  }

  cgpu_physical_device_limits device_limits;
  c_result = cgpu_get_physical_device_limits(device, &device_limits);
  gatling_cgpu_ensure(c_result);

  /* Map scene file for copying. */
  gatling_file* scene_file;
  const bool ok = gatling_file_open(options.input_file, GATLING_FILE_USAGE_READ, &scene_file);
  if (!ok) {
    gatling_fail("Unable to read scene file.");
  }

  uint64_t scene_data_size = gatling_file_size(scene_file);
  uint8_t* mapped_scene_data = (uint8_t*) gatling_mmap(
    scene_file,
    0,
    scene_data_size
  );

  if (!mapped_scene_data) {
    gatling_fail("Unable to map scene file.");
  }

  /* Create input and output buffers. */
  const uint64_t path_seg_struct_size = 32;
  const uint64_t hit_info_struct_size = 32;
  const uint64_t path_seg_header_size = 16;
  const uint64_t hit_info_header_size = 16;
  const uint64_t node_struct_size = 32;

  const struct file_header {
    uint64_t node_buf_offset;
    uint64_t node_buf_size;
    uint64_t face_buf_offset;
    uint64_t face_buf_size;
    uint64_t vertex_buf_offset;
    uint64_t vertex_buf_size;
    uint64_t material_buf_offset;
    uint64_t material_buf_size;
    float aabb_min_x;
    float aabb_min_y;
    float aabb_min_z;
    float aabb_max_x;
    float aabb_max_y;
    float aabb_max_z;
  } file_header = *((struct file_header*) &mapped_scene_data[0]);

  uint64_t device_buf_size = 0;
  const uint64_t offset_align = device_limits.minStorageBufferOffsetAlignment;

  const uint64_t new_node_buf_offset = gatling_align_buffer(offset_align, file_header.node_buf_size, &device_buf_size);
  const uint64_t new_face_buf_offset = gatling_align_buffer(offset_align, file_header.face_buf_size, &device_buf_size);
  const uint64_t new_vertex_buf_offset = gatling_align_buffer(offset_align, file_header.vertex_buf_size, &device_buf_size);
  const uint64_t new_material_buf_offset = gatling_align_buffer(offset_align, file_header.material_buf_size, &device_buf_size);

  const uint64_t prim_ray_count = options.image_width * options.image_height * options.spp;

  const uint64_t path_segment_buf_offset = 0;
  const uint64_t path_segment_buf_size = prim_ray_count * path_seg_struct_size + path_seg_header_size;
  const uint64_t hit_info_buf_offset = path_segment_buf_offset + path_segment_buf_size;
  const uint64_t hit_info_buf_size = prim_ray_count * hit_info_struct_size + hit_info_header_size;

  const uint64_t intermediate_buf_size = path_segment_buf_size + hit_info_buf_size;

  const uint64_t output_buffer_size = options.image_width * options.image_height * sizeof(float) * 4;

  cgpu_buffer staging_buffer_in;
  cgpu_buffer input_buffer;
  cgpu_buffer intermediate_buffer;
  cgpu_buffer output_buffer;
  cgpu_buffer staging_buffer_out;
  cgpu_buffer timestamp_buffer;

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
    device_buf_size,
    &staging_buffer_in
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    device_buf_size,
    &input_buffer
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    intermediate_buf_size,
    &intermediate_buffer
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
    CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL,
    output_buffer_size,
    &output_buffer
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
    output_buffer_size,
    &staging_buffer_out
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_create_buffer(
    device,
    CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST,
    CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_COHERENT |
      CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
    32 * sizeof(uint64_t),
    &timestamp_buffer
  );
  gatling_cgpu_ensure(c_result);

  uint8_t* mapped_staging_mem;
  c_result = cgpu_map_buffer(
    device,
    staging_buffer_in,
    0,
    CGPU_WHOLE_SIZE,
    (void*)&mapped_staging_mem
  );
  gatling_cgpu_ensure(c_result);

  memcpy(&mapped_staging_mem[new_node_buf_offset],
         &mapped_scene_data[file_header.node_buf_offset], file_header.node_buf_size);
  memcpy(&mapped_staging_mem[new_face_buf_offset],
         &mapped_scene_data[file_header.face_buf_offset], file_header.face_buf_size);
  memcpy(&mapped_staging_mem[new_vertex_buf_offset],
         &mapped_scene_data[file_header.vertex_buf_offset], file_header.vertex_buf_size);
  memcpy(&mapped_staging_mem[new_material_buf_offset],
         &mapped_scene_data[file_header.material_buf_offset], file_header.material_buf_size);

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

  /* Unmap the scene data since it's been copied. */
  gatling_munmap(scene_file, mapped_scene_data);
  gatling_file_close(scene_file);

  const uint32_t num_shader_resource_buffers = 7;
  cgpu_shader_resource_buffer shader_resource_buffers[] = {
    { 0,       output_buffer,                       0,               CGPU_WHOLE_SIZE },
    { 1, intermediate_buffer, path_segment_buf_offset,         path_segment_buf_size },
    { 2,        input_buffer,     new_node_buf_offset,     file_header.node_buf_size },
    { 3,        input_buffer,     new_face_buf_offset,     file_header.face_buf_size },
    { 4,        input_buffer,   new_vertex_buf_offset,   file_header.vertex_buf_size },
    { 5,        input_buffer, new_material_buf_offset, file_header.material_buf_size },
    { 6, intermediate_buffer,     hit_info_buf_offset,             hit_info_buf_size }
  };

  char dir_path[1024];
  gatling_get_parent_directory(argv[0], dir_path);

  char kernel_ray_gen_shader_path[1152];
  snprintf(kernel_ray_gen_shader_path, 1152, "%s/shaders/kernel_ray_gen.comp.spv", dir_path);
  char kernel_extend_shader_path[1152];
  snprintf(kernel_extend_shader_path, 1152, "%s/shaders/kernel_extend.comp.spv", dir_path);
  char kernel_shade_shader_path[1152];
  snprintf(kernel_shade_shader_path, 1152, "%s/shaders/kernel_shade.comp.spv", dir_path);

  gatling_pipeline pipeline_ray_gen;
  gatling_pipeline pipeline_extend;
  gatling_pipeline pipeline_shade;

  {
    const float aabb_length[3] = {
      file_header.aabb_max_x - file_header.aabb_min_x,
      file_header.aabb_max_y - file_header.aabb_min_y,
      file_header.aabb_max_z - file_header.aabb_min_z
    };

    const float camera_target[3] = {
      (file_header.aabb_max_x + file_header.aabb_min_x) * 0.5f,
      (file_header.aabb_max_y + file_header.aabb_min_y) * 0.5f,
      (file_header.aabb_max_z + file_header.aabb_min_z) * 0.5f
    };
    const float camera_origin[3] = {
      camera_target[0] + aabb_length[0],
      camera_target[1] + aabb_length[1],
      camera_target[2] + aabb_length[2]
    };
    const float camera_fov = 0.872665f;

    const cgpu_specialization_constant speccs[] = {
      { .constant_id =  0, .p_data = (void*) &device_limits.subgroupSize, .size = 4 },
      { .constant_id =  1, .p_data = (void*) &device_limits.subgroupSize, .size = 4 },
      { .constant_id =  2, .p_data = (void*) &options.spp,                .size = 4 },
      { .constant_id =  3, .p_data = (void*) &options.image_width,        .size = 4 },
      { .constant_id =  4, .p_data = (void*) &options.image_height,       .size = 4 },
      { .constant_id =  5, .p_data = (void*) &camera_origin[0],           .size = 4 },
      { .constant_id =  6, .p_data = (void*) &camera_origin[1],           .size = 4 },
      { .constant_id =  7, .p_data = (void*) &camera_origin[2],           .size = 4 },
      { .constant_id =  8, .p_data = (void*) &camera_target[0],           .size = 4 },
      { .constant_id =  9, .p_data = (void*) &camera_target[1],           .size = 4 },
      { .constant_id = 10, .p_data = (void*) &camera_target[2],           .size = 4 },
      { .constant_id = 11, .p_data = (void*) &camera_fov,                 .size = 4 }
    };

    gatling_create_pipeline(
      device,
      kernel_ray_gen_shader_path,
      shader_resource_buffers,
      num_shader_resource_buffers,
      12,
      speccs,
      &pipeline_ray_gen
    );
  }

  {
    /* Let's just hardcode this. In the future, we can calculate how many nodes
     * we need (log8(node_count) * 2) and how many we can fit into shared memory
     * (via device_limits.maxComputeSharedMemorySize and our workgroup size). */
    const uint32_t traversal_stack_size = 6;
    const uint32_t sm_traversal_stack_size = 12;

    const cgpu_specialization_constant speccs[] = {
      { .constant_id = 0, .p_data = (void*) &device_limits.subgroupSize, .size = 4 },
      { .constant_id = 1, .p_data = (void*) &options.spp,                .size = 4 },
      { .constant_id = 2, .p_data = (void*) &options.image_width,        .size = 4 },
      { .constant_id = 3, .p_data = (void*) &options.image_height,       .size = 4 },
      { .constant_id = 4, .p_data = (void*) &traversal_stack_size,       .size = 4 },
      { .constant_id = 5, .p_data = (void*) &sm_traversal_stack_size,    .size = 4 }
    };

    gatling_create_pipeline(
      device,
      kernel_extend_shader_path,
      shader_resource_buffers,
      num_shader_resource_buffers,
      6,
      speccs,
      &pipeline_extend
    );
  }

  {
    const cgpu_specialization_constant speccs[] = {
      { .constant_id = 0, .p_data = (void*) &device_limits.subgroupSize, .size = 4 }
    };

    gatling_create_pipeline(
      device,
      kernel_shade_shader_path,
      shader_resource_buffers,
      num_shader_resource_buffers,
      1,
      speccs,
      &pipeline_shade
    );
  }

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
  buffer_memory_barrier_1.offset = 0;
  buffer_memory_barrier_1.size = CGPU_WHOLE_SIZE;

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
    (options.image_width / device_limits.subgroupSize) + 1,
    (options.image_height / device_limits.subgroupSize) + 1,
    1
  );
  gatling_cgpu_ensure(c_result);

  c_result = cgpu_cmd_write_timestamp(command_buffer, 2u);
  gatling_cgpu_ensure(c_result);

  /* Trace rays. */
  cgpu_buffer_memory_barrier buffer_memory_barrier_2;
  buffer_memory_barrier_2.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_2.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_2.buffer = intermediate_buffer;
  buffer_memory_barrier_2.offset = path_segment_buf_offset;
  buffer_memory_barrier_2.size = path_segment_buf_size;

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
    ((options.image_width * options.image_height * options.spp) / device_limits.subgroupSize) + 1,
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
  buffer_memory_barrier_3.buffer = intermediate_buffer;
  buffer_memory_barrier_3.offset = hit_info_buf_offset;
  buffer_memory_barrier_3.size = hit_info_buf_size;

  cgpu_buffer_memory_barrier buffer_memory_barrier_4;
  buffer_memory_barrier_4.src_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_WRITE;
  buffer_memory_barrier_4.dst_access_flags = CGPU_MEMORY_ACCESS_FLAG_SHADER_READ;
  buffer_memory_barrier_4.buffer = output_buffer;
  buffer_memory_barrier_4.offset = 0u;
  buffer_memory_barrier_4.size = CGPU_WHOLE_SIZE;

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
    ((options.image_width * options.image_height * options.spp) / device_limits.subgroupSize) + 1,
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
  buffer_memory_barrier_5.offset = 0;
  buffer_memory_barrier_5.size = CGPU_WHOLE_SIZE;

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
  float* image_data = malloc(output_buffer_size);

  c_result = cgpu_map_buffer(
    device,
    staging_buffer_out,
    0,
    CGPU_WHOLE_SIZE,
    (void**)&mapped_staging_mem
  );
  gatling_cgpu_ensure(c_result);

  memcpy(
    image_data,
    mapped_staging_mem,
    output_buffer_size
  );

  c_result = cgpu_unmap_buffer(
    device,
    staging_buffer_out
  );
  gatling_cgpu_ensure(c_result);

  /* Save image. */
  gatling_save_img(
    image_data,
    output_buffer_size / 4,
    options.image_width,
    options.image_height,
    options.output_file
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
  c_result = cgpu_destroy_buffer(device, intermediate_buffer);
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
