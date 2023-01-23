#include "common.glsl"

layout(binding = 0, std430) buffer PixelsBuffer { vec4 pixels[]; };

layout(binding = 1, std430) readonly buffer FacesBuffer { face faces[]; };

#ifdef NEXT_EVENT_ESTIMATION
layout(binding = 2, std430) readonly buffer EmissiveFacesBuffer { uint emissive_face_indices[]; };
#endif

layout(binding = 3, std430) readonly buffer VerticesBuffer { fvertex vertices[]; };

#if defined(HAS_TEXTURES_2D) || defined(HAS_TEXTURES_3D)
layout(binding = 4) uniform sampler tex_sampler;
#endif

#ifdef HAS_TEXTURES_2D
layout(binding = 5) uniform texture2D textures_2d[TEXTURE_COUNT_2D];
#endif

#ifdef HAS_TEXTURES_3D
layout(binding = 6) uniform texture3D textures_3d[TEXTURE_COUNT_3D];
#endif

layout(binding = 7) uniform accelerationStructureEXT sceneAS;

layout(push_constant) uniform PCBuffer {
    vec3  CAMERA_POSITION;
    uint  IMAGE_WIDTH;
    vec3  CAMERA_FORWARD;
    uint  IMAGE_HEIGHT;
    vec3  CAMERA_UP;
    float CAMERA_VFOV;
    vec4  BACKGROUND_COLOR;
    uint  SAMPLE_COUNT;
    uint  MAX_BOUNCES;
    float MAX_SAMPLE_VALUE;
    uint  RR_BOUNCE_OFFSET;
    float RR_INV_MIN_TERM_PROB;
    uint  SAMPLE_OFFSET;
} PC;
