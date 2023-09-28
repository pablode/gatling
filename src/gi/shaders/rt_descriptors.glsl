#include "interface/rp_main.h"
#include "common.glsl"

layout(binding = BINDING_INDEX_OUT_PIXELS, std430) buffer Framebuffer { vec4 pixels[]; };

layout(binding = BINDING_INDEX_FACES, std430) readonly buffer FacesBuffer { Face faces[]; };

#if SPHERE_LIGHT_COUNT > 0
layout(binding = BINDING_INDEX_SPHERE_LIGHTS, std430) readonly buffer SphereLightBuffer { SphereLight sphereLights[]; };
#endif

#if DISTANT_LIGHT_COUNT > 0
layout(binding = BINDING_INDEX_DISTANT_LIGHTS, std430) readonly buffer DistantLightBuffer { DistantLight distantLights[]; };
#endif

#if RECT_LIGHT_COUNT > 0
layout(binding = BINDING_INDEX_RECT_LIGHTS, std430) readonly buffer RectLightBuffer { RectLight rectLights[]; };
#endif

layout(binding = BINDING_INDEX_VERTICES, std430) readonly buffer VerticesBuffer { FVertex vertices[]; };

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

layout(push_constant) uniform PushConstantBlock { PushConstants PC; };
