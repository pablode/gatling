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

struct UniformData
{
  GI_VEC4 domeLightRotation;
  GI_VEC3 domeLightEmissionMultiplier;
  GI_UINT domeLightDiffuseSpecularPacked;
  GI_UINT maxTextureIndex;
  GI_UINT sphereLightCount;
  GI_UINT distantLightCount;
  GI_UINT rectLightCount;
  GI_UINT diskLightCount;
  GI_UINT totalLightCount;
  GI_FLOAT metersPerSceneUnit;
  GI_UINT  maxVolumeWalkLength; // NOTE: can be quantized
  GI_VEC3  cameraPosition;
  GI_UINT  imageDims;
  GI_VEC3  cameraForward;
  GI_FLOAT focusDistance;
  GI_VEC3  cameraUp;
  GI_FLOAT cameraVFoV;
  GI_UINT  sampleOffset;
  GI_FLOAT lensRadius;
  GI_UINT  spp;
  GI_FLOAT invSampleCount;
  GI_FLOAT maxSampleValue;
  GI_UINT  maxBouncesAndRrBounceOffset;
  GI_FLOAT rrInvMinTermProb;
  GI_FLOAT lightIntensityMultiplier;
  GI_UINT  clipRangePacked;
  GI_FLOAT sensorExposure;
};

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

const GI_UINT BLAS_PAYLOAD_BITFLAG_FLIP_FACING = (1 << 0);
const GI_UINT BLAS_PAYLOAD_BITFLAG_DOUBLE_SIDED = (1 << 1);

struct BlasPayload
{
  GI_UINT64 bufferAddress;
  GI_UINT   vertexOffset;
  GI_UINT   bitfield;
};

const GI_UINT FACE_ID_MASK = 0x3FFFFFFF;
const GI_UINT FACE_ID_STRIDE_MASK = 0xC0000000;
const GI_UINT FACE_ID_STRIDE_OFFSET = 30;

const GI_UINT SCENE_DATA_INVALID = 0xFFFFFFFFu;
const GI_UINT SCENE_DATA_ALIGNMENT = 32; // must be equal or larger largest type
const GI_UINT SCENE_DATA_OFFSET_MASK = 0x0FFFFFFFu; // (1 << 28) - 1
const GI_UINT SCENE_DATA_STRIDE_MASK = 0x30000000u; // 0011 0000...
const GI_UINT SCENE_DATA_STRIDE_OFFSET = 28;
const GI_UINT SCENE_DATA_INTERPOLATION_MASK = 0xC0000000u; // 1100 0000 ...
const GI_UINT SCENE_DATA_INTERPOLATION_OFFSET = 30; // bits

const GI_UINT MAX_SCENE_DATA_COUNT = 6;
const GI_UINT MAX_TEXTURE_COUNT = 65535;

struct BlasPayloadBufferPreamble
{
  GI_INT objectId;
  GI_UINT faceIdsInfo;
  GI_UINT sceneDataInfos[MAX_SCENE_DATA_COUNT];
};
#ifdef __cplusplus
static_assert((sizeof(BlasPayloadBufferPreamble) % 32) == 0);
#endif

// set 0
GI_BINDING_INDEX(UNIFORM_DATA,    0)
GI_BINDING_INDEX(SPHERE_LIGHTS,   1)
GI_BINDING_INDEX(DISTANT_LIGHTS,  2)
GI_BINDING_INDEX(RECT_LIGHTS,     3)
GI_BINDING_INDEX(DISK_LIGHTS,     4)
GI_BINDING_INDEX(SAMPLER,         5)
GI_BINDING_INDEX(SCENE_AS,        8)
GI_BINDING_INDEX(BLAS_PAYLOADS,   9)
GI_BINDING_INDEX(INSTANCE_IDS,   10)

GI_BINDING_INDEX(AOV_CLEAR_VALUES_F, 11)
GI_BINDING_INDEX(AOV_CLEAR_VALUES_I, 12)

GI_BINDING_INDEX(AOV_COLOR,        13)
GI_BINDING_INDEX(AOV_NORMAL,       14)
GI_BINDING_INDEX(AOV_NEE,          15)
GI_BINDING_INDEX(AOV_BARYCENTRICS, 16)
GI_BINDING_INDEX(AOV_TEXCOORDS,    17)
GI_BINDING_INDEX(AOV_BOUNCES,      18)
GI_BINDING_INDEX(AOV_CLOCK_CYCLES, 19)
GI_BINDING_INDEX(AOV_OPACITY,      20)
GI_BINDING_INDEX(AOV_TANGENTS,     21)
GI_BINDING_INDEX(AOV_BITANGENTS,   22)
GI_BINDING_INDEX(AOV_THIN_WALLED,  23)
GI_BINDING_INDEX(AOV_OBJECT_ID,    24)
GI_BINDING_INDEX(AOV_DEPTH,        25)
GI_BINDING_INDEX(AOV_FACE_ID,      26)
GI_BINDING_INDEX(AOV_INSTANCE_ID,  27)
GI_BINDING_INDEX(AOV_DOUBLE_SIDED, 28)
GI_BINDING_INDEX(AOV_ALBEDO,       29)
GI_BINDING_INDEX(AOV_OIDN,         30)

// set 1 & set 2 (alised array)
GI_BINDING_INDEX(TEXTURES,     0)

GI_INTERFACE_END()

#endif
