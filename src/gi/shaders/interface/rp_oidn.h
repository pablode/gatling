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

#ifdef __cplusplus
const GI_INT CONV_IMPL_SEQ = 0;
const GI_INT CONV_IMPL_SHMEM = 1;
#else
#define CONV_IMPL_SEQ 0
#define CONV_IMPL_SHMEM 1
#endif

struct UniformData
{
  GI_UINT imageWidth;
  GI_UINT imageHeight;
  GI_UINT weightOffset;
  GI_UINT biasOffset;
};

// set 0
GI_BINDING_INDEX(UNIFORM_DATA, 0)
GI_BINDING_INDEX(INPUT_BUF1, 1)
GI_BINDING_INDEX(OUTPUT_BUF, 2)
GI_BINDING_INDEX(TENSOR_BUF, 3)
GI_BINDING_INDEX(INPUT_BUF2, 4)
GI_BINDING_INDEX(VALUE_SCALE_BUF, 5)

GI_INTERFACE_END()

#endif
