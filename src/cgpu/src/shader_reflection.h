#ifndef CGPU_SHADER_REFLECTION_H
#define CGPU_SHADER_REFLECTION_H

#include "cgpu.h"

typedef enum CgpuShaderReflectionResourceType {
  CGPU_SHADER_REFLECTION_RESOURCE_TYPE_BUFFER,
  CGPU_SHADER_REFLECTION_RESOURCE_TYPE_STORAGE_IMAGE,
  CGPU_SHADER_REFLECTION_RESOURCE_TYPE_SAMPLED_IMAGE,
  CGPU_SHADER_REFLECTION_RESOURCE_TYPE_PUSH_CONSTANT
} CgpuShaderReflectionResourceType;

typedef struct cgpu_shader_reflection_resource {
  uint32_t binding;
  CgpuShaderReflectionResourceType resource_type;
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
