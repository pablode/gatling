#include "common.glsl"

struct ShadeRayPayload
{
    /* inout */ f16vec3 throughput;
    /* inout */ uint16_t bitfield; // Bitfield values:
                                   // 1xxxxxxxxxxxxxxx bool inside
                                   // x111111111111111 uint bounce
    /* inout */ f16vec3 radiance;
    /* - */     uint16_t padding;
#ifdef RAND_4D
    /* inout */ uvec4 rng_state;
#else
    /* inout */ uint rng_state;
#endif
    /* out */   vec3 ray_origin;
    /* out */   vec3 ray_dir;
    /* out */   vec3 neeOrigin;
    /* out */   vec3 neeLightPos;
    /* out */   vec3 neeContrib;
};

struct ShadowRayPayload
{
#ifdef RAND_4D
    /* inout */ uvec4 rng_state;
#else
    /* inout */ uint rng_state;
#endif
    /* out */   bool shadowed;
};

const int PAYLOAD_INDEX_SHADE = 0;
const int PAYLOAD_INDEX_SHADOW = 1;
