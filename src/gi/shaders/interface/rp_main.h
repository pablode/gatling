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

struct FVertex
{
  /* f32 pos[3], f32 bsign */
  SI_VEC4 field1;
  /* u32 norm, u32 tan, f32 texcoords[2] */
  SI_VEC4 field2;
};

struct Face
{
  SI_UINT v_0;
  SI_UINT v_1;
  SI_UINT v_2;
};

struct SphereLight
{
  SI_VEC3  pos;
  SI_FLOAT radius;
  SI_VEC3  baseEmission;
  SI_FLOAT padding;
};

struct DistantLight
{
  SI_VEC3  direction;
  SI_FLOAT angle;
  SI_VEC3  baseEmission;
  SI_FLOAT padding;
};

struct RectLight
{
  SI_VEC3  origin;
  SI_FLOAT width;
  SI_VEC3  baseEmission;
  SI_FLOAT height;
  SI_VEC3  direction;
  SI_FLOAT padding;
};

struct PushConstants
{
  SI_VEC3  cameraPosition;
  SI_UINT  imageDims;
  SI_VEC3  cameraForward;
  SI_FLOAT focusDistance;
  SI_VEC3  cameraUp;
  SI_FLOAT cameraVFoV;
  SI_VEC4  backgroundColor;
  SI_UINT  sampleOffset;
  SI_FLOAT lensRadius;
  SI_UINT  sampleCount;
  SI_FLOAT maxSampleValue;
  SI_VEC4  domeLightRotation;
  SI_VEC3  domeLightEmissionMultiplier;
  SI_UINT  maxBouncesAndRrBounceOffset;
  SI_FLOAT rrInvMinTermProb;
  SI_FLOAT lightIntensityMultiplier;
  /* 2 floats free */
};

SI_BINDING_INDEX(OUT_PIXELS,     0)
SI_BINDING_INDEX(FACES,          1)
SI_BINDING_INDEX(SPHERE_LIGHTS,  2)
SI_BINDING_INDEX(DISTANT_LIGHTS, 3)
SI_BINDING_INDEX(RECT_LIGHTS,    4)
SI_BINDING_INDEX(VERTICES,       5)
SI_BINDING_INDEX(SAMPLER,        6)
SI_BINDING_INDEX(TEXTURES_2D,    7)
SI_BINDING_INDEX(TEXTURES_3D,    8)
SI_BINDING_INDEX(SCENE_AS,       9)

SI_NAMESPACE_END()

#endif
