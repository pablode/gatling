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

#pragma once

#include <stdint.h>
#include <vector>

namespace gtl
{
  struct CgpuShaderReflectionBinding
  {
    uint32_t binding; // TODO: rename to index
    uint32_t count;
    int descriptorType;
    bool readAccess;
    bool writeAccess;
    uint32_t dim;
  };

  struct CgpuShaderReflectionDescriptorSet
  {
    std::vector<CgpuShaderReflectionBinding> bindings;
  };

  struct CgpuShaderReflection
  {
    std::vector<CgpuShaderReflectionDescriptorSet> descriptorSets;
    uint32_t pushConstantsSize;
    uint32_t maxRayPayloadSize;
    uint32_t maxRayHitAttributeSize;
    uint32_t workgroupSize[3];
  };

  bool cgpuReflectShader(const uint32_t* spv, uint64_t size, CgpuShaderReflection* reflection);
}
