#include "common.glsl"

struct ShadeRayPayload
{
    /* inout */ vec3 throughput;
    /* inout */ uint bitfield; // MSB: inside/outside. rest: bounce count
    /* inout */ vec3 radiance;
    /* inout */ RNG_STATE_TYPE rng_state;
    /* out */   vec3 ray_origin;
    /* out */   vec3 ray_dir;
    /* out */   vec3 neeToLight;
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
