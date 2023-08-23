//
// Copyright (C) 2023 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "shaderReflection.h"

#include <smallVector.h>
#include <volk.h>
#include <spirv_reflect.h>
#include <stdlib.h>
#include <assert.h>

using namespace gtl;

bool cgpuReflectShader(const uint32_t* spv, uint64_t size, CgpuShaderReflection* reflection)
{
  SpvReflectShaderModule shaderModule = {};
  SpvReflectModuleFlags flags = SPV_REFLECT_MODULE_FLAG_NO_COPY;
  if (spvReflectCreateShaderModule2(flags, size, spv, &shaderModule) != SPV_REFLECT_RESULT_SUCCESS)
  {
    return false;
  }

  bool result = false;
  GbSmallVector<SpvReflectDescriptorBinding*, 32> bindings;

  uint32_t bindingCount;
  if (spvReflectEnumerateDescriptorBindings(&shaderModule, &bindingCount, nullptr) != SPV_REFLECT_RESULT_SUCCESS)
  {
    goto fail;
  }

  if (bindingCount > 0)
  {
    bindings.resize(bindingCount);

    if (spvReflectEnumerateDescriptorBindings(&shaderModule, &bindingCount, bindings.data()) != SPV_REFLECT_RESULT_SUCCESS)
    {
      goto fail;
    }

    reflection->bindings.resize(bindingCount);

    for (uint32_t i = 0; i < bindings.size(); i++)
    {
      const SpvReflectDescriptorBinding* srcBinding = bindings[i];

      CgpuShaderReflectionBinding* dstBinding = &reflection->bindings[i];
      dstBinding->binding = srcBinding->binding;
      dstBinding->count = srcBinding->count;
      dstBinding->descriptorType = (int)srcBinding->descriptor_type;
      // Unfortunately SPIRV-Reflect lacks the functionality to detect read accesses:
      // https://github.com/KhronosGroup/SPIRV-Reflect/issues/99
      dstBinding->readAccess = srcBinding->accessed;
      const SpvReflectTypeDescription* typeDescription = srcBinding->type_description;
      dstBinding->writeAccess = srcBinding->accessed && ~(typeDescription->decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE);
    }
  }

  if (shaderModule.push_constant_block_count == 0)
  {
    reflection->pushConstantsSize = 0;
  }
  else
  {
    assert(shaderModule.push_constant_block_count == 1);

    const SpvReflectBlockVariable* pcBlock = &shaderModule.push_constant_blocks[0];
    reflection->pushConstantsSize = pcBlock->size;
  }

  result = true;

fail:
  spvReflectDestroyShaderModule(&shaderModule);
  return result;
}
