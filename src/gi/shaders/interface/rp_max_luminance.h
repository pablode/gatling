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

#ifndef RP_MAX_LUMINANCE_H
#define RP_MAX_LUMINANCE_H

#include "interface/gtl.h"

GI_INTERFACE_BEGIN(rp_max_luminance)

const GI_UINT WG_SIZE_X = 32;
const GI_UINT WG_SIZE_Y = 32;

struct PushConstants
{
  GI_UINT imageWidth;
  GI_UINT imageHeight;
};

// set 0
GI_BINDING_INDEX(INPUT_BUF, 0)
GI_BINDING_INDEX(OUTPUT_BUF, 1)

GI_INTERFACE_END()

#endif
