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

#ifndef RP_OIDN_H
#define RP_OIDN_H

#include "interface/gtl.h"

GI_INTERFACE_BEGIN(rp_oidn)

const GI_UINT WG_SIZE_X = 16;
const GI_UINT WG_SIZE_Y = 16;

struct PushConstants
{
  GI_UINT imageWidth;
  GI_UINT imageHeight;
  GI_UINT weightOffset; // offset of kernel into the weights buffer
  GI_UINT biasOffset;
};

// set 0
//GI_BINDING_INDEX(SCENE_PARAMS,    0)
//GI_BINDING_INDEX(SPHERE_LIGHTS,   1)

GI_INTERFACE_END()

#endif
