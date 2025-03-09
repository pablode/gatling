#include "interface/rp_main.h"
#include "common.glsl"

layout(binding = BINDING_INDEX_SCENE_PARAMS, std430) readonly buffer SceneParamsBuffer { SceneParams sceneParams; };

layout(binding = BINDING_INDEX_SPHERE_LIGHTS, std430) readonly buffer SphereLightBuffer { SphereLight sphereLights[]; };
layout(binding = BINDING_INDEX_DISTANT_LIGHTS, std430) readonly buffer DistantLightBuffer { DistantLight distantLights[]; };
layout(binding = BINDING_INDEX_RECT_LIGHTS, std430) readonly buffer RectLightBuffer { RectLight rectLights[]; };
layout(binding = BINDING_INDEX_DISK_LIGHTS, std430) readonly buffer DiskLightBuffer { DiskLight diskLights[]; };

layout(binding = BINDING_INDEX_SAMPLER) uniform sampler tex_sampler;

layout(binding = BINDING_INDEX_TEXTURES_2D) uniform texture2D textures_2d[MAX_TEXTURE_COUNT];
layout(binding = BINDING_INDEX_TEXTURES_3D) uniform texture3D textures_3d[MAX_TEXTURE_COUNT];

layout(binding = BINDING_INDEX_SCENE_AS) uniform accelerationStructureEXT sceneAS;

layout(binding = BINDING_INDEX_BLAS_PAYLOADS, std430) readonly buffer BlasPayloadBuffer {
  BlasPayload blas_payloads[];
};

layout(binding = BINDING_INDEX_INSTANCE_IDS, std430) readonly buffer InstanceIdsBuffer { int InstanceIds[]; };

layout(binding = BINDING_INDEX_AOV_CLEAR_VALUES_F, std430) readonly buffer ClearValueBufferF { vec4 ClearValuesF[]; };
layout(binding = BINDING_INDEX_AOV_CLEAR_VALUES_I, std430) readonly buffer ClearValueBufferI { ivec4 ClearValuesI[]; };

#if (AOV_MASK & AOV_BIT_COLOR) != 0
layout(binding = BINDING_INDEX_AOV_COLOR, std430) buffer Framebuffer { vec4 ColorAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_NORMAL) != 0
layout(binding = BINDING_INDEX_AOV_NORMAL, std430) writeonly buffer NormalBuffer { vec3 NormalsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_NEE) != 0
layout(binding = BINDING_INDEX_AOV_NEE, std430) writeonly writeonly buffer NeeBuffer { vec3 NeeAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BARYCENTRICS) != 0
layout(binding = BINDING_INDEX_AOV_BARYCENTRICS, std430) writeonly buffer BarycentricsBuffer { vec3 BarycentricsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_TEXCOORDS) != 0
layout(binding = BINDING_INDEX_AOV_TEXCOORDS, std430) writeonly buffer TexcoordsBuffer { vec3 TexcoordsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BOUNCES) != 0
layout(binding = BINDING_INDEX_AOV_BOUNCES, std430) writeonly buffer BouncesBuffer { vec3 BouncesAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_CLOCK_CYCLES) != 0
layout(binding = BINDING_INDEX_AOV_CLOCK_CYCLES, std430) writeonly buffer ClockCyclesBuffer { uvec3 ClockCyclesAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_OPACITY) != 0
layout(binding = BINDING_INDEX_AOV_OPACITY, std430) writeonly buffer OpacityBuffer { vec3 OpacityAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_TANGENTS) != 0
layout(binding = BINDING_INDEX_AOV_TANGENTS, std430) writeonly buffer TangentsBuffer { vec3 TangentsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BITANGENTS) != 0
layout(binding = BINDING_INDEX_AOV_BITANGENTS, std430) writeonly buffer BitangentsBuffer { vec3 BitangentsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_THIN_WALLED) != 0
layout(binding = BINDING_INDEX_AOV_THIN_WALLED, std430) writeonly buffer ThinWalledBuffer { vec3 ThinWalledAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_OBJECT_ID) != 0
layout(binding = BINDING_INDEX_AOV_OBJECT_ID, std430) writeonly buffer ObjectIdBuffer { int ObjectIdAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEPTH) != 0
layout(binding = BINDING_INDEX_AOV_DEPTH, std430) writeonly buffer DepthBuffer { float DepthAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_FACE_ID) != 0
layout(binding = BINDING_INDEX_AOV_FACE_ID, std430) writeonly buffer FaceIdBuffer { int FaceIdAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_INSTANCE_ID) != 0
layout(binding = BINDING_INDEX_AOV_INSTANCE_ID, std430) writeonly buffer InstanceIdBuffer { int InstanceIdAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_DOUBLE_SIDED) != 0
layout(binding = BINDING_INDEX_AOV_DOUBLE_SIDED, std430) writeonly buffer DoubleSidedBuffer { vec3 DoubleSidedAov[]; };
#endif

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type (see below) */) buffer IndexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  Face data[];
};

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type: vertex */) buffer VertexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  FVertex data[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer RawIntBuffer { int data[]; };

layout(push_constant) uniform PushConstantBlock { PushConstants PC; };
