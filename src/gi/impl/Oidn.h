//
// Copyright (C) 2025 Pablo Delgado Krämer
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

#include <stdbool.h>
#include <stddef.h>

#include <gtl/cgpu/Cgpu.h>

#include "Tza.h"

namespace gtl
{
  class GiGlslShaderGen;
  class GgpuDelayedResourceDestroyer;
  class GgpuStager;

  struct GiOidnState;

  GiOidnState* giOidnCreateState(CgpuDevice device,
                                 GiGlslShaderGen& shaderGen,
                                 GgpuStager& stager,
                                 GgpuDelayedResourceDestroyer& resourceDestroyer,
                                 const GiTzaTensorDescriptions& tensorDescriptions,
                                 const uint8_t* tensorData);

  void giOidnDestroyState(GiOidnState* state);

  bool giOidnUpdateState(GiOidnState* state, CgpuDevice device, uint32_t imageWidth, uint32_t imageHeight);

  CgpuBuffer giOidnGetInputBuffer(GiOidnState* state);
  CgpuBuffer giOidnGetOutputBuffer(GiOidnState* state);

  void giOidnRender(GiOidnState* state, CgpuCommandBuffer commandBuffer);
}
