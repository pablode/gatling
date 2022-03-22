#include "shader_reflection.h"

#include <volk.h>
#include <spirv_reflect.h>
#include <stdlib.h>
#include <assert.h>

bool cgpu_perform_shader_reflection(uint64_t size,
                                    const uint32_t* p_spv,
                                    cgpu_shader_reflection* p_reflection)
{
  SpvReflectShaderModule shader_module = {0};
  if (spvReflectCreateShaderModule(size, p_spv, &shader_module) != SPV_REFLECT_RESULT_SUCCESS)
  {
    return false;
  }

  SpvReflectDescriptorBinding** bindings = NULL;
  p_reflection->resources = NULL;
  bool result = false;

  uint32_t binding_count;
  if (spvReflectEnumerateDescriptorBindings(&shader_module, &binding_count, NULL) != SPV_REFLECT_RESULT_SUCCESS)
  {
    goto fail;
  }
  assert(binding_count > 0);

  bindings = malloc(binding_count * sizeof(SpvReflectDescriptorBinding*));
  if (spvReflectEnumerateDescriptorBindings(&shader_module, &binding_count, bindings) != SPV_REFLECT_RESULT_SUCCESS)
  {
    goto fail;
  }

  p_reflection->resource_count = 0;
  p_reflection->resources = (cgpu_shader_reflection_resource*) malloc(sizeof(cgpu_shader_reflection_resource) * binding_count);

  for (uint32_t i = 0; i < binding_count; i++)
  {
    const SpvReflectDescriptorBinding* binding = bindings[i];

    if (!binding->accessed)
    {
      continue;
    }

    cgpu_shader_reflection_resource* sr_res = &p_reflection->resources[p_reflection->resource_count++];
    sr_res->descriptor_type = (int) binding->descriptor_type;

    // Unfortunately SPIRV-Reflect lacks this functionality:
    // https://github.com/KhronosGroup/SPIRV-Reflect/issues/99
    const SpvReflectTypeDescription* type_description = binding->type_description;
    sr_res->write_access = ~(type_description->decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE);
    sr_res->read_access = true;

    sr_res->binding = binding->binding;
  }

  assert(shader_module.push_constant_block_count == 1);
  const SpvReflectBlockVariable* pc_block = &shader_module.push_constant_blocks[0];
  p_reflection->push_constants_size = pc_block->size;

  result = true;

fail:
  if (!result)
  {
    free(p_reflection->resources);
  }
  free(bindings);
  spvReflectDestroyShaderModule(&shader_module);
  return result;
}

void cgpu_destroy_shader_reflection(cgpu_shader_reflection* p_reflection)
{
  free(p_reflection->resources);
}
