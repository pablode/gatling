#include "common.glsl"

struct RayPayload
{
#ifdef RAND_4D
    uvec4 rng_state;
#else
    uint rng_state;
#endif
    vec3 ray_origin;
    vec3 ray_dir;
    vec3 throughput;
    vec3 radiance;
    // Bitfield values:
    // 1xxxxxxxxxxxxxxx bool inside
    // x111111111111111 uint bounce
    uint bitfield;
};

struct ShadowRayPayload
{
#ifdef RAND_4D
    uvec4 rng_state;
#else
    uint rng_state;
#endif
    bool shadowed;
};
