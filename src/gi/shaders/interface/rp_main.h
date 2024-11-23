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

#ifndef RP_MAIN_H
#define RP_MAIN_H

#include "interface/gtl.h"

GI_INTERFACE_BEGIN(rp_main)

struct FVertex
{
  /* f32 pos[3], f32 bsign */
  GI_VEC4 field1;
  /* u32 norm, u32 tan, f32 texcoords[2] */
  GI_VEC4 field2;
};

struct Face
{
  GI_UINT v_0;
  GI_UINT v_1;
  GI_UINT v_2;
};

struct SphereLight
{
  GI_VEC3  pos;
  GI_UINT  diffuseSpecularPacked;
  GI_VEC3  baseEmission;
  GI_FLOAT area;
  GI_VEC3  radiusXYZ;
  GI_FLOAT padding;
};

struct DistantLight
{
  GI_VEC3  direction;
  GI_FLOAT angle;
  GI_VEC3  baseEmission;
  GI_UINT  diffuseSpecularPacked;
  GI_VEC3  padding;
  GI_FLOAT invPdf;
};

struct RectLight
{
  GI_VEC3  origin;
  GI_FLOAT width;
  GI_VEC3  baseEmission;
  GI_FLOAT height;
  GI_UVEC2 tangentFramePacked;
  GI_UINT  diffuseSpecularPacked;
  GI_FLOAT padding;
};

struct DiskLight
{
  GI_VEC3  origin;
  GI_FLOAT radiusX;
  GI_VEC3  baseEmission;
  GI_FLOAT radiusY;
  GI_UVEC2 tangentFramePacked;
  GI_UINT  diffuseSpecularPacked;
  GI_FLOAT padding;
};

struct PushConstants
{
  GI_VEC3  cameraPosition;
  GI_UINT  imageDims;
  GI_VEC3  cameraForward;
  GI_FLOAT focusDistance;
  GI_VEC3  cameraUp;
  GI_FLOAT cameraVFoV;
  GI_UINT  sampleOffset;
  GI_FLOAT lensRadius;
  GI_UINT  sampleCount;
  GI_FLOAT maxSampleValue;
  GI_VEC4  domeLightRotation;
  GI_VEC3  domeLightEmissionMultiplier;
  GI_UINT  domeLightDiffuseSpecularPacked;
  GI_UINT  maxBouncesAndRrBounceOffset;
  GI_FLOAT rrInvMinTermProb;
  GI_FLOAT lightIntensityMultiplier;
  GI_UINT  clipRangePacked;
  GI_FLOAT sensorExposure;
  GI_UINT  maxVolumeWalkLength; // NOTE: can be quantized
  /* 2 floats free */
};

const GI_UINT BLAS_PAYLOAD_BITFLAG_FLIP_FACING = (1 << 0);

struct BlasPayload
{
  GI_UINT64 bufferAddress;
  GI_UINT   vertexOffset;
  GI_UINT   bitfield;
};

const GI_UINT BLAS_PREAMBLE_SCENE_DATA_BITFLAG_CONSTANT = 0x80000000u; //(1 << 31);
const GI_UINT BLAS_PREAMBLE_SCENE_DATA_OFFSET_MASK = 0x7FFFFFFFu;

const GI_UINT MAX_SCENE_DATA_COUNT = 7;

struct BlasPayloadBufferPreamble
{
  GI_INT objectId;
  GI_UINT sceneDataInfos[MAX_SCENE_DATA_COUNT];
};

GI_BINDING_INDEX(OUT_PIXELS,     0)
GI_BINDING_INDEX(SPHERE_LIGHTS,  1)
GI_BINDING_INDEX(DISTANT_LIGHTS, 2)
GI_BINDING_INDEX(RECT_LIGHTS,    3)
GI_BINDING_INDEX(DISK_LIGHTS,    4)
GI_BINDING_INDEX(SAMPLER,        5)
GI_BINDING_INDEX(TEXTURES_2D,    6)
GI_BINDING_INDEX(TEXTURES_3D,    7)
GI_BINDING_INDEX(SCENE_AS,       8)
GI_BINDING_INDEX(BLAS_PAYLOADS,  9)

GI_INTERFACE_END()

#endif
