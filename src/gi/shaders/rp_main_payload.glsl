#include "common.glsl"

#define SHADE_RAY_PAYLOAD_VOLUME_WALK_MISS_FLAG 0x40000000u
#define SHADE_RAY_PAYLOAD_MEDIUM_IDX_MASK 0x0f000000u
#define SHADE_RAY_PAYLOAD_MEDIUM_IDX_OFFSET 24
#define SHADE_RAY_PAYLOAD_WALK_MASK 0x00fff000u
#define SHADE_RAY_PAYLOAD_WALK_OFFSET 12
#define SHADE_RAY_PAYLOAD_BOUNCES_MASK 0x00000fffu
#define SHADE_RAY_PAYLOAD_TERMINATE_FLAG 0x80000000u

struct Medium
{
    vec3 ior;
    vec3 sigma_s;
    vec3 sigma_t; // sigma_a + sigma_s
    float bias;
};

// NOTE: size needs to match _GetRpMainMaxRayPayloadSize()
struct ShadeRayPayload
{
    /* inout */ vec3 throughput;

    /*               1000 0000 0000 0000 0000 0000 0000 0000 terminate
     *               0100 0000 0000 0000 0000 0000 0000 0000 volume walk miss
     *               0011 0000 0000 0000 0000 0000 0000 0000 unused
     *               0000 1111 0000 0000 0000 0000 0000 0000 medium index [0, 256)
     *               0000 0000 1111 1111 1111 0000 0000 0000 walk length [0, 4096)
     *               0000 0000 0000 0000 0000 1111 1111 1111 bounces [0, 4096) */
    /* inout */ uint bitfield;

    /* inout */ vec3 radiance;

    /* inout */ RNG_STATE_TYPE rng_state;

#if MEDIUM_STACK_SIZE > 0
    /* inout */ Medium media[MEDIUM_STACK_SIZE];
    /* inout */ vec3 walkSegmentPdf;
#endif

    /* out */   vec3 ray_origin;
    /* out */   vec3 ray_dir;
    /* out */   vec3 neeToLight;
    /* out */   vec3 neeContrib;

    /* in  */   uint pixelIndex;
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

#if MEDIUM_STACK_SIZE > 0
void shadeRayPayloadIncrementWalk(inout ShadeRayPayload payload)
{
    uint bitfield = payload.bitfield;
    bitfield &= SHADE_RAY_PAYLOAD_WALK_MASK;
    bitfield = min(bitfield + 1, SHADE_RAY_PAYLOAD_WALK_MASK);

    payload.bitfield &= ~SHADE_RAY_PAYLOAD_WALK_MASK;
    payload.bitfield |= bitfield;
}

uint shadeRayPayloadGetWalk(in ShadeRayPayload payload)
{
    return (payload.bitfield & SHADE_RAY_PAYLOAD_WALK_MASK) >> SHADE_RAY_PAYLOAD_WALK_OFFSET;
}
#endif

uint shadeRayPayloadGetMediumIdx(in ShadeRayPayload payload)
{
    uint bitfield = payload.bitfield;
    bitfield &= SHADE_RAY_PAYLOAD_MEDIUM_IDX_MASK;
    bitfield >>= SHADE_RAY_PAYLOAD_MEDIUM_IDX_OFFSET;

    // The medium index can exceed the max stack size. This is used to ensure an
    // equal number of stack pushes/pops when the number of nested media exceeds
    // the max stack size. The case is only relevant when we push to the stack;
    // for common use we want to choose the uppermost/current medium.
    uint maxMediumIdx = max(1, MEDIUM_STACK_SIZE);

    return min(bitfield, maxMediumIdx);
}

void shadeRayPayloadSetMediumIdx(inout ShadeRayPayload payload, uint idx)
{
    payload.bitfield &= ~SHADE_RAY_PAYLOAD_MEDIUM_IDX_MASK;
    payload.bitfield |= min(idx << SHADE_RAY_PAYLOAD_MEDIUM_IDX_OFFSET, SHADE_RAY_PAYLOAD_MEDIUM_IDX_MASK);
}

const int PAYLOAD_INDEX_SHADE = 0;
const int PAYLOAD_INDEX_SHADOW = 1;
