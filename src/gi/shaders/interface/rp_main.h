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

#ifndef SI_RP_MAIN
#define SI_RP_MAIN

#include "interface/gtl.h"

SI_NAMESPACE_BEGIN(rp_main)

#define RP_MAIN_PUSH_CONSTANT_LIST    \
{                                     \
  SI_VEC3  CAMERA_POSITION;           \
  SI_UINT  IMAGE_WIDTH;               \
  SI_VEC3  CAMERA_FORWARD;            \
  SI_UINT  IMAGE_HEIGHT;              \
  SI_VEC3  CAMERA_UP;                 \
  SI_FLOAT CAMERA_VFOV;               \
  SI_VEC4  BACKGROUND_COLOR;          \
  SI_UINT  SAMPLE_COUNT;              \
  SI_UINT  MAX_BOUNCES;               \
  SI_FLOAT MAX_SAMPLE_VALUE;          \
  SI_UINT  RR_BOUNCE_OFFSET;          \
  SI_VEC3  DOMELIGHT_TRANSFORM_COL0;  \
  SI_FLOAT RR_INV_MIN_TERM_PROB;      \
  SI_VEC3  DOMELIGHT_TRANSFORM_COL1;  \
  SI_UINT  SAMPLE_OFFSET;             \
  SI_VEC3  DOMELIGHT_TRANSFORM_COL2;  \
}
#ifdef __cplusplus
struct PushConstants RP_MAIN_PUSH_CONSTANT_LIST;
#endif

SI_BINDING_INDEX(OUT_PIXELS,     0)
SI_BINDING_INDEX(FACES,          1)
SI_BINDING_INDEX(EMISSIVE_FACES, 2)
SI_BINDING_INDEX(VERTICES,       3)
SI_BINDING_INDEX(SAMPLER,        4)
SI_BINDING_INDEX(TEXTURES_2D,    5)
SI_BINDING_INDEX(TEXTURES_3D,    6)
SI_BINDING_INDEX(SCENE_AS,       7)

SI_NAMESPACE_END()

#endif
