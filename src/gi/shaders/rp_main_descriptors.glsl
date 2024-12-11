#include "interface/rp_main.h"
#include "common.glsl"

#if SPHERE_LIGHT_COUNT > 0
layout(binding = BINDING_INDEX_SPHERE_LIGHTS, std430) readonly buffer SphereLightBuffer { SphereLight sphereLights[]; };
#endif

#if DISTANT_LIGHT_COUNT > 0
layout(binding = BINDING_INDEX_DISTANT_LIGHTS, std430) readonly buffer DistantLightBuffer { DistantLight distantLights[]; };
#endif

#if RECT_LIGHT_COUNT > 0
layout(binding = BINDING_INDEX_RECT_LIGHTS, std430) readonly buffer RectLightBuffer { RectLight rectLights[]; };
#endif

#if DISK_LIGHT_COUNT > 0
layout(binding = BINDING_INDEX_DISK_LIGHTS, std430) readonly buffer DiskLightBuffer { DiskLight diskLights[]; };
#endif

#if (TEXTURE_COUNT_2D > 0) || (TEXTURE_COUNT_3D > 0)
layout(binding = BINDING_INDEX_SAMPLER) uniform sampler tex_sampler;
#endif

#if TEXTURE_COUNT_2D > 0
layout(binding = BINDING_INDEX_TEXTURES_2D) uniform texture2D textures_2d[TEXTURE_COUNT_2D];
#endif

#if TEXTURE_COUNT_3D > 0
layout(binding = BINDING_INDEX_TEXTURES_3D) uniform texture3D textures_3d[TEXTURE_COUNT_3D];
#endif

layout(binding = BINDING_INDEX_SCENE_AS) uniform accelerationStructureEXT sceneAS;

layout(binding = BINDING_INDEX_BLAS_PAYLOADS, std430) buffer BlasPayloadBuffer {
  BlasPayload blas_payloads[];
};

layout(binding = BINDING_INDEX_AOV_CLEAR_VALUES_F, std430) buffer ClearValueBufferF { vec4 ClearValuesF[]; };
layout(binding = BINDING_INDEX_AOV_CLEAR_VALUES_I, std430) buffer ClearValueBufferI { ivec4 ClearValuesI[]; };

#if (AOV_MASK & AOV_BIT_COLOR) != 0
layout(binding = BINDING_INDEX_AOV_COLOR, std430) buffer Framebuffer { vec4 ColorAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_NORMAL) != 0
layout(binding = BINDING_INDEX_AOV_NORMAL, std430) buffer NormalBuffer { vec3 NormalsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_NEE) != 0
layout(binding = BINDING_INDEX_AOV_NEE, std430) buffer NeeBuffer { vec3 NeeAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BARYCENTRICS) != 0
layout(binding = BINDING_INDEX_AOV_BARYCENTRICS, std430) buffer BarycentricsBuffer { vec3 BarycentricsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_TEXCOORDS) != 0
layout(binding = BINDING_INDEX_AOV_TEXCOORDS, std430) buffer TexcoordsBuffer { vec3 TexcoordsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BOUNCES) != 0
layout(binding = BINDING_INDEX_AOV_BOUNCES, std430) buffer BouncesBuffer { vec3 BouncesAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_CLOCK_CYCLES) != 0
layout(binding = BINDING_INDEX_AOV_CLOCK_CYCLES, std430) buffer ClockCyclesBuffer { uint ClockCyclesAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_OPACITY) != 0
layout(binding = BINDING_INDEX_AOV_OPACITY, std430) buffer OpacityBuffer { vec3 OpacityAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_TANGENTS) != 0
layout(binding = BINDING_INDEX_AOV_TANGENTS, std430) buffer TangentsBuffer { vec3 TangentsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_BITANGENTS) != 0
layout(binding = BINDING_INDEX_AOV_BITANGENTS, std430) buffer BitangentsBuffer { vec3 BitangentsAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_DEBUG_THIN_WALLED) != 0
layout(binding = BINDING_INDEX_AOV_THIN_WALLED, std430) buffer ThinWalledBuffer { vec3 ThinWalledAov[]; };
#endif
#if (AOV_MASK & AOV_BIT_OBJECTID) != 0
layout(binding = BINDING_INDEX_AOV_OBJECT_ID, std430) buffer ObjectIdBuffer { int ObjectIdAov[]; };
#endif

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type (see below) */) buffer IndexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  Face data[];
};

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type: vertex */) buffer VertexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  FVertex data[];
};

layout(push_constant) uniform PushConstantBlock { PushConstants PC; };
