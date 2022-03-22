#ifndef CGPU_SHADER_REFLECTION_H
#define CGPU_SHADER_REFLECTION_H

#include "cgpu.h"

typedef struct cgpu_shader_reflection_resource {
  uint32_t binding;
  int descriptor_type;
  bool write_access;
  bool read_access;
} cgpu_shader_reflection_resource;

typedef struct cgpu_shader_reflection {
  uint32_t push_constants_size;
  uint32_t resource_count;
  cgpu_shader_reflection_resource* resources;
} cgpu_shader_reflection;

bool cgpu_perform_shader_reflection(
  uint64_t size,
  const uint32_t* p_spv,
  cgpu_shader_reflection* p_reflection
);

void cgpu_destroy_shader_reflection(
  cgpu_shader_reflection* p_reflection
);

#endif
