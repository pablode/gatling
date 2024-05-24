#include "common.glsl"

#define SHADE_RAY_PAYLOAD_MEDIUM_IDX_MASK 0x0f000000u
#define SHADE_RAY_PAYLOAD_MEDIUM_IDX_OFFSET 24
#define SHADE_RAY_PAYLOAD_WALK_MASK 0x00fff000u
#define SHADE_RAY_PAYLOAD_WALK_OFFSET 12
#define SHADE_RAY_PAYLOAD_BOUNCES_MASK 0x00000fffu
#define SHADE_RAY_PAYLOAD_TERMINATE_FLAG 0x80000000u

struct ShadeRayPayload
{
    /* inout */ vec3 throughput;

    /*               1000 0000 0000 0000 0000 0000 0000 0000 terminate
     *               0111 0000 0000 0000 0000 0000 0000 0000 unused
     *               0000 1111 0000 0000 0000 0000 0000 0000 medium index [0, 256)
     *               0000 0000 1111 1111 1111 0000 0000 0000 walk length [0, 4096)
     *               0000 0000 0000 0000 0000 1111 1111 1111 bounces [0, 4096) */
    /* inout */ uint bitfield;

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
