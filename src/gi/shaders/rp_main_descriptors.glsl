#include "interface/rp_main.h"
#include "common.glsl"

layout(binding = BINDING_INDEX_UNIFORM_DATA, std140) uniform UniformDataUbo { UniformData ubo; };
layout(binding = BINDING_INDEX_SPHERE_LIGHTS, std430) readonly buffer SphereLightBuffer { SphereLight sphereLights[]; };
layout(binding = BINDING_INDEX_DISTANT_LIGHTS, std430) readonly buffer DistantLightBuffer { DistantLight distantLights[]; };
layout(binding = BINDING_INDEX_RECT_LIGHTS, std430) readonly buffer RectLightBuffer { RectLight rectLights[]; };
layout(binding = BINDING_INDEX_DISK_LIGHTS, std430) readonly buffer DiskLightBuffer { DiskLight diskLights[]; };

layout(binding = BINDING_INDEX_SAMPLER) uniform sampler tex_sampler;

layout(binding = BINDING_INDEX_TEXTURE_INFOS, std430) readonly buffer TextureInfoBuffer { TextureInfo textureInfos[]; };

layout(binding = BINDING_INDEX_SCENE_AS) uniform accelerationStructureEXT sceneAS;

layout(binding = BINDING_INDEX_BLAS_PAYLOADS, std430) readonly buffer BlasPayloadBuffer {
  BlasPayload blas_payloads[];
};

layout(binding = BINDING_INDEX_INSTANCE_IDS, std430) readonly buffer InstanceIdsBuffer { int InstanceIds[]; };

layout(binding = BINDING_INDEX_AOV_CLEAR_VALUES_F, std430) readonly buffer ClearValueBufferF { vec4 ClearValuesF[]; };
layout(binding = BINDING_INDEX_AOV_CLEAR_VALUES_I, std430) readonly buffer ClearValueBufferI { ivec4 ClearValuesI[]; };

// AOVs are storage images so that they can be consumed by Hydra without a
// CPU copy. Writes to multi-component AOVs use rgba32f as there is no rgb32f
// storage image format.
#if (AOV_MASK & AOV_BIT_COLOR) != 0
layout(binding = BINDING_INDEX_AOV_COLOR, rgba32f) uniform image2D ColorAov;
#endif
#if (AOV_MASK & AOV_BIT_NORMAL) != 0
layout(binding = BINDING_INDEX_AOV_NORMAL, rgba32f) uniform image2D NormalsAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_NEE) != 0
layout(binding = BINDING_INDEX_AOV_NEE, rgba32f) uniform writeonly image2D NeeAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BARYCENTRICS) != 0
layout(binding = BINDING_INDEX_AOV_BARYCENTRICS, rgba32f) uniform writeonly image2D BarycentricsAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_TEXCOORDS) != 0
layout(binding = BINDING_INDEX_AOV_TEXCOORDS, rgba32f) uniform writeonly image2D TexcoordsAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BOUNCES) != 0
layout(binding = BINDING_INDEX_AOV_BOUNCES, rgba32f) uniform writeonly image2D BouncesAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_CLOCK_CYCLES) != 0
layout(binding = BINDING_INDEX_AOV_CLOCK_CYCLES, rgba32f) uniform writeonly image2D ClockCyclesAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_OPACITY) != 0
layout(binding = BINDING_INDEX_AOV_OPACITY, rgba32f) uniform writeonly image2D OpacityAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_TANGENTS) != 0
layout(binding = BINDING_INDEX_AOV_TANGENTS, rgba32f) uniform writeonly image2D TangentsAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BITANGENTS) != 0
layout(binding = BINDING_INDEX_AOV_BITANGENTS, rgba32f) uniform writeonly image2D BitangentsAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_THIN_WALLED) != 0
layout(binding = BINDING_INDEX_AOV_THIN_WALLED, rgba32f) uniform writeonly image2D ThinWalledAov;
#endif
#if (AOV_MASK & AOV_BIT_OBJECT_ID) != 0
layout(binding = BINDING_INDEX_AOV_OBJECT_ID, r32i) uniform writeonly iimage2D ObjectIdAov;
#endif
#if (AOV_MASK & AOV_BIT_DEPTH) != 0
layout(binding = BINDING_INDEX_AOV_DEPTH, r32f) uniform writeonly image2D DepthAov;
#endif
#if (AOV_MASK & AOV_BIT_FACE_ID) != 0
layout(binding = BINDING_INDEX_AOV_FACE_ID, r32i) uniform writeonly iimage2D FaceIdAov;
#endif
#if (AOV_MASK & AOV_BIT_INSTANCE_ID) != 0
layout(binding = BINDING_INDEX_AOV_INSTANCE_ID, r32i) uniform writeonly iimage2D InstanceIdAov;
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_DOUBLE_SIDED) != 0
layout(binding = BINDING_INDEX_AOV_DOUBLE_SIDED, rgba32f) uniform writeonly image2D DoubleSidedAov;
#endif
#if (AOV_MASK & AOV_BIT_ALBEDO) != 0
layout(binding = BINDING_INDEX_AOV_ALBEDO, rgba32f) uniform image2D AlbedoAov;
#endif

ivec2 aovImageCoord(uint pixelIndex)
{
  uint imageWidth = ubo.imageDims & 0xFFFFu;
  return ivec2(pixelIndex % imageWidth, pixelIndex / imageWidth);
}

layout(set = 1, binding = BINDING_INDEX_TEXTURES) uniform texture2D textures_2d[MAX_TEXTURE_COUNT];

layout(set = 2, binding = BINDING_INDEX_TEXTURES) uniform texture3D textures_3d[MAX_TEXTURE_COUNT];

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type (see below) */) buffer IndexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  Face data[];
};

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type: vertex */) buffer VertexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  FVertex data[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer RawIntBuffer { int data[]; };
