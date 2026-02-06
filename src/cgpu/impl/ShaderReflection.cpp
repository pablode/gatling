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

#include "ShaderReflection.h"

#include <spirv_reflect.h>
#include <stdlib.h>
#include <assert.h>

namespace gtl
{
  static uint32_t cgpuGetTypeDescriptionSize(const SpvReflectTypeDescription& typeDesc);

  static uint32_t cgpuGetStructSize(const SpvReflectTypeDescription& typeDesc)
  {
      uint32_t size = 0;

      for (uint32_t i = 0; i < typeDesc.member_count; i++)
      {
        size += cgpuGetTypeDescriptionSize(typeDesc.members[i]);
      }

      return size;
  }

  static uint32_t cgpuGetTypeDescriptionSize(const SpvReflectTypeDescription& typeDesc)
  {
    const SpvReflectNumericTraits& nt = typeDesc.traits.numeric;

    uint32_t byteWidth = 8;
    uint32_t size = (nt.scalar.width / byteWidth);
    if (nt.vector.component_count > 0)
    {
      size *= nt.vector.component_count;
    }

    if (typeDesc.struct_type_description)
    {
      uint32_t elemSize = cgpuGetStructSize(*typeDesc.struct_type_description);

      const auto& array = typeDesc.traits.array;
      for (uint32_t d = 0; d < array.dims_count; d++)
      {
        size += elemSize * array.dims[d];
      }
    }

    return size;
  }

  static uint32_t cgpuGetInterfaceVarSize(const SpvReflectInterfaceVariable* var)
  {
    uint32_t size = 0;

    for (uint32_t i = 0; i < var->member_count; i++)
    {
      size += cgpuGetInterfaceVarSize(&var->members[i]);
    }

    if (var->type_description)
    {
      size += cgpuGetTypeDescriptionSize(*var->type_description);
    }

    return size;
  };

  bool cgpuReflectShader(const uint32_t* spv, uint64_t size, CgpuShaderReflection* reflection)
  {
    SpvReflectShaderModule shaderModule = {};
    SpvReflectModuleFlags flags = SPV_REFLECT_MODULE_FLAG_NO_COPY;
    if (spvReflectCreateShaderModule2(flags, size, spv, &shaderModule) != SPV_REFLECT_RESULT_SUCCESS)
    {
      return false;
    }

    bool result = false;

    uint32_t interfaceVarCount;
    std::vector<SpvReflectInterfaceVariable*> interfaceVars;

    uint32_t descriptorSetCount;
    std::vector<SpvReflectDescriptorSet*> descriptorSets;

    if (spvReflectEnumerateInterfaceVariables(&shaderModule, &interfaceVarCount, nullptr) != SPV_REFLECT_RESULT_SUCCESS)
    {
      goto fail;
    }

    interfaceVars.resize(interfaceVarCount);
    if (spvReflectEnumerateInterfaceVariables(&shaderModule, &interfaceVarCount, interfaceVars.data()) != SPV_REFLECT_RESULT_SUCCESS)
    {
      goto fail;
    }

    reflection->maxRayPayloadSize = 0;
    reflection->maxRayHitAttributeSize = 0;
    reflection->payloadCount = 0;

    for (const SpvReflectInterfaceVariable* interfaceVar : interfaceVars)
    {
      if (interfaceVar->storage_class == SpvStorageClassRayPayloadKHR ||
          interfaceVar->storage_class == SpvStorageClassIncomingRayPayloadKHR)
      {
        uint32_t size = cgpuGetInterfaceVarSize(interfaceVar);

        if (size > reflection->maxRayPayloadSize)
        {
          reflection->maxRayPayloadSize = size;
        }

        reflection->payloadCount++;
      }
      else if (interfaceVar->storage_class == SpvStorageClassHitAttributeKHR)
      {
        uint32_t size = cgpuGetInterfaceVarSize(interfaceVar);

        if (size > reflection->maxRayHitAttributeSize)
        {
          reflection->maxRayHitAttributeSize = size;
        }
      }
    }

    if (spvReflectEnumerateDescriptorSets(&shaderModule, &descriptorSetCount, nullptr) != SPV_REFLECT_RESULT_SUCCESS)
    {
      goto fail;
    }

    descriptorSets.resize(descriptorSetCount);
    if (spvReflectEnumerateDescriptorSets(&shaderModule, &descriptorSetCount, descriptorSets.data()) != SPV_REFLECT_RESULT_SUCCESS)
    {
      goto fail;
    }

    reflection->descriptorSets.resize(descriptorSetCount);
    for (uint32_t j = 0; j < descriptorSetCount; j++)
    {
      const SpvReflectDescriptorSet* srcDescriptorSet = descriptorSets[j];
      uint32_t bindingCount = srcDescriptorSet->binding_count;

      if (bindingCount ==  0)
      {
        continue;
      }

      CgpuShaderReflectionDescriptorSet& dstDescriptorSet = reflection->descriptorSets[j];
      std::vector<CgpuShaderReflectionBinding>& dstBindings = dstDescriptorSet.bindings;
      dstBindings.resize(bindingCount);

      for (uint32_t i = 0; i < bindingCount; i++)
      {
        const SpvReflectDescriptorBinding* srcBinding = srcDescriptorSet->bindings[i];

        CgpuShaderReflectionBinding& dstBinding = dstBindings[i];
        dstBinding.binding = srcBinding->binding;
        dstBinding.count = srcBinding->count;
        dstBinding.descriptorType = (int)srcBinding->descriptor_type;
        // Unfortunately SPIRV-Reflect lacks the functionality to detect read accesses:
        // https://github.com/KhronosGroup/SPIRV-Reflect/issues/99
        dstBinding.readAccess = srcBinding->accessed;
        const SpvReflectTypeDescription* typeDescription = srcBinding->type_description;
        dstBinding.writeAccess = srcBinding->accessed && ~(typeDescription->decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE);
        dstBinding.dim = uint32_t(srcBinding->image.dim) + 1; // enum starts with 0
      }
    }

    if (shaderModule.entry_point_count != 1)
    {
      goto fail;
    }
    else
    {
      const SpvReflectEntryPoint& entryPoint = shaderModule.entry_points[0];
      reflection->workgroupSize[0] = entryPoint.local_size.x;
      reflection->workgroupSize[1] = entryPoint.local_size.y;
      reflection->workgroupSize[2] = entryPoint.local_size.z;
    }

    result = true;

fail:
    spvReflectDestroyShaderModule(&shaderModule);
    return result;
  }
}
