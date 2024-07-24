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
#include <gtl/gb/SmallVector.h>

namespace gtl
{
  struct CgpuShaderReflectionBinding
  {
    uint32_t binding;
    uint32_t count;
    int descriptorType;
    bool readAccess;
    bool writeAccess;
  };

  struct CgpuShaderReflection
  {
    uint32_t pushConstantsSize;
    GbSmallVector<CgpuShaderReflectionBinding, 32> bindings;
  };

  bool cgpuReflectShader(const uint32_t* spv, uint64_t size, CgpuShaderReflection* reflection);
}
