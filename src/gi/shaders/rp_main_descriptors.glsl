#include "interface/rp_main.h"
#include "common.glsl"

layout(binding = BINDING_INDEX_OUT_PIXELS, std430) buffer Framebuffer { vec4 pixels[]; };

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

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type (see below) */) buffer IndexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  Face data[];
};

layout(buffer_reference, std430, buffer_reference_align = 32/* largest type: vertex */) buffer VertexBuffer {
  BlasPayloadBufferPreamble preamble; // important: preamble size must match alignment
  FVertex data[];
};

layout(push_constant) uniform PushConstantBlock { PushConstants PC; };
