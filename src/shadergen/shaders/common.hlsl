const static float FLOAT_MAX = 3.402823466e38;
const static float FLOAT_MIN = 1.175494351e-38;
const static float PI = 3.1415926535897932384626433832795;
const static float TRI_EPS = 0.000000001;

// These must match the IDs in gi.h
#define AOV_ID_COLOR 0
#define AOV_ID_NORMAL 1
#define AOV_ID_DEBUG_NEE 2
#define AOV_ID_DEBUG_BVH_STEPS 3
#define AOV_ID_DEBUG_TRI_TESTS 4
#define AOV_ID_DEBUG_BARYCENTRICS 5
#define AOV_ID_DEBUG_TEXCOORDS 6
#define AOV_ID_DEBUG_BOUNCES 7

struct fvertex
{
    /* pos.{x, y, z}, tex.u */
    float4 field1;
    /* norm.{x, y, z}, tex.v */
    float4 field2;
};

struct face
{
    uint v_0;
    uint v_1;
    uint v_2;
    uint mat_idx;
};

#ifdef BVH_ENABLED
struct bvh_node
{
    uint4 f1;
    uint4 f2;
    uint4 f3;
    uint4 f4;
    uint4 f5;
};
#endif

struct RayInfo
{
    float3 origin;
    float  tmax;
    float3 dir;
    float  padding;
};

struct Hit_info
{
    uint face_idx;
    float2 bc;
#if AOV_ID == AOV_ID_DEBUG_BVH_STEPS
    uint bvh_steps;
#elif AOV_ID == AOV_ID_DEBUG_TRI_TESTS
    uint tri_tests;
#endif
};

// RNG producing on a four-element vector.
// From: "Hash Functions for GPU Rendering" by Jarzynski and Olano.
// Licensed under CC BY-ND 3.0: https://creativecommons.org/licenses/by-nd/3.0/
uint4 pcg4d(uint4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

uint4 pcg4d_init(uint2 pixel_coords, uint frame_num)
{
    return uint4(pixel_coords.xy, frame_num, 0);
}

float4 uint4ToFloat4(uint4 v)
{
    v >>= 9;
    v |= 0x3f800000;
    return float4(asfloat(v)) - float4(1.0, 1.0, 1.0, 1.0);
}

float4 pcg4d_next(inout uint4 rng_state)
{
    rng_state.w++;
    return uint4ToFloat4(pcg4d(rng_state));
}

// From: "A Fast and Robust Method for Avoiding Self-Intersection"
// Wächter and Binder, Ch. 6 in Ray Tracing Gems I.
float3 offset_ray_origin(float3 p, float3 geom_normal)
{
    const float origin = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale = 256.0;

    int3 of_i = int3(
        int_scale * geom_normal.x,
        int_scale * geom_normal.y,
        int_scale * geom_normal.z
    );

    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0.0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0.0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0.0) ? -of_i.z : of_i.z))
    );

    return float3(
        abs(p.x) < origin ? (p.x + float_scale * geom_normal.x) : p_i.x,
        abs(p.y) < origin ? (p.y + float_scale * geom_normal.y) : p_i.y,
        abs(p.z) < origin ? (p.z + float_scale * geom_normal.z) : p_i.z
    );
}

// Hand-crafted Morton code lookup table; index corresponds to thread-index in group.
static uint MORTON_2D_LUT_32x8[256] = {
    ( 0 << 16) | 0, ( 1 << 16) | 0, ( 0 << 16) | 1, ( 1 << 16) | 1, ( 2 << 16) | 0, ( 3 << 16) | 0, ( 2 << 16) | 1, ( 3 << 16) | 1,
    ( 0 << 16) | 2, ( 1 << 16) | 2, ( 0 << 16) | 3, ( 1 << 16) | 3, ( 2 << 16) | 2, ( 3 << 16) | 2, ( 2 << 16) | 3, ( 3 << 16) | 3,
    ( 4 << 16) | 0, ( 5 << 16) | 0, ( 4 << 16) | 1, ( 5 << 16) | 1, ( 6 << 16) | 0, ( 7 << 16) | 0, ( 6 << 16) | 1, ( 7 << 16) | 1,
    ( 4 << 16) | 2, ( 5 << 16) | 2, ( 4 << 16) | 3, ( 5 << 16) | 3, ( 6 << 16) | 2, ( 7 << 16) | 2, ( 6 << 16) | 3, ( 7 << 16) | 3,
    ( 0 << 16) | 4, ( 1 << 16) | 4, ( 0 << 16) | 5, ( 1 << 16) | 5, ( 2 << 16) | 4, ( 3 << 16) | 4, ( 2 << 16) | 5, ( 3 << 16) | 5,
    ( 0 << 16) | 6, ( 1 << 16) | 6, ( 0 << 16) | 7, ( 1 << 16) | 7, ( 2 << 16) | 6, ( 3 << 16) | 6, ( 2 << 16) | 7, ( 3 << 16) | 7,
    ( 4 << 16) | 4, ( 5 << 16) | 4, ( 4 << 16) | 5, ( 5 << 16) | 5, ( 6 << 16) | 4, ( 7 << 16) | 4, ( 6 << 16) | 5, ( 7 << 16) | 5,
    ( 4 << 16) | 6, ( 5 << 16) | 6, ( 4 << 16) | 7, ( 5 << 16) | 7, ( 6 << 16) | 6, ( 7 << 16) | 6, ( 6 << 16) | 7, ( 7 << 16) | 7,
    ( 8 << 16) | 0, ( 9 << 16) | 0, ( 8 << 16) | 1, ( 9 << 16) | 1, (10 << 16) | 0, (11 << 16) | 0, (10 << 16) | 1, (11 << 16) | 1,
    ( 8 << 16) | 2, ( 9 << 16) | 2, ( 8 << 16) | 3, ( 9 << 16) | 3, (10 << 16) | 2, (11 << 16) | 2, (10 << 16) | 3, (11 << 16) | 3,
    (12 << 16) | 0, (13 << 16) | 0, (12 << 16) | 1, (13 << 16) | 1, (14 << 16) | 0, (15 << 16) | 0, (14 << 16) | 1, (15 << 16) | 1,
    (12 << 16) | 2, (13 << 16) | 2, (12 << 16) | 3, (13 << 16) | 3, (14 << 16) | 2, (15 << 16) | 2, (14 << 16) | 3, (15 << 16) | 3,
    ( 8 << 16) | 4, ( 9 << 16) | 4, ( 8 << 16) | 5, ( 9 << 16) | 5, (10 << 16) | 4, (11 << 16) | 4, (10 << 16) | 5, (11 << 16) | 5,
    ( 8 << 16) | 6, ( 9 << 16) | 6, ( 8 << 16) | 7, ( 9 << 16) | 7, (10 << 16) | 6, (11 << 16) | 6, (10 << 16) | 7, (11 << 16) | 7,
    (12 << 16) | 4, (13 << 16) | 4, (12 << 16) | 5, (13 << 16) | 5, (14 << 16) | 4, (15 << 16) | 4, (14 << 16) | 5, (15 << 16) | 5,
    (12 << 16) | 6, (13 << 16) | 6, (12 << 16) | 7, (13 << 16) | 7, (14 << 16) | 6, (15 << 16) | 6, (14 << 16) | 7, (15 << 16) | 7,
    (16 << 16) | 0, (17 << 16) | 0, (16 << 16) | 1, (17 << 16) | 1, (18 << 16) | 0, (19 << 16) | 0, (18 << 16) | 1, (19 << 16) | 1,
    (16 << 16) | 2, (17 << 16) | 2, (16 << 16) | 3, (17 << 16) | 3, (18 << 16) | 2, (19 << 16) | 2, (18 << 16) | 3, (19 << 16) | 3,
    (20 << 16) | 0, (21 << 16) | 0, (20 << 16) | 1, (21 << 16) | 1, (22 << 16) | 0, (23 << 16) | 0, (22 << 16) | 1, (23 << 16) | 1,
    (20 << 16) | 2, (21 << 16) | 2, (20 << 16) | 3, (21 << 16) | 3, (22 << 16) | 2, (23 << 16) | 2, (22 << 16) | 3, (23 << 16) | 3,
    (16 << 16) | 4, (17 << 16) | 4, (16 << 16) | 5, (17 << 16) | 5, (18 << 16) | 4, (19 << 16) | 4, (18 << 16) | 5, (19 << 16) | 5,
    (16 << 16) | 6, (17 << 16) | 6, (16 << 16) | 7, (17 << 16) | 7, (18 << 16) | 6, (19 << 16) | 6, (18 << 16) | 7, (19 << 16) | 7,
    (20 << 16) | 4, (21 << 16) | 4, (20 << 16) | 5, (21 << 16) | 5, (22 << 16) | 4, (23 << 16) | 4, (22 << 16) | 5, (23 << 16) | 5,
    (20 << 16) | 6, (21 << 16) | 6, (20 << 16) | 7, (21 << 16) | 7, (22 << 16) | 6, (23 << 16) | 6, (22 << 16) | 7, (23 << 16) | 7,
    (24 << 16) | 0, (25 << 16) | 0, (24 << 16) | 1, (25 << 16) | 1, (26 << 16) | 0, (27 << 16) | 0, (26 << 16) | 1, (27 << 16) | 1,
    (24 << 16) | 2, (25 << 16) | 2, (24 << 16) | 3, (25 << 16) | 3, (26 << 16) | 2, (27 << 16) | 2, (26 << 16) | 3, (27 << 16) | 3,
    (28 << 16) | 0, (29 << 16) | 0, (28 << 16) | 1, (29 << 16) | 1, (30 << 16) | 0, (31 << 16) | 0, (30 << 16) | 1, (31 << 16) | 1,
    (28 << 16) | 2, (29 << 16) | 2, (28 << 16) | 3, (29 << 16) | 3, (30 << 16) | 2, (31 << 16) | 2, (30 << 16) | 3, (31 << 16) | 3,
    (24 << 16) | 4, (25 << 16) | 4, (24 << 16) | 5, (25 << 16) | 5, (26 << 16) | 4, (27 << 16) | 4, (26 << 16) | 5, (27 << 16) | 5,
    (24 << 16) | 6, (25 << 16) | 6, (24 << 16) | 7, (25 << 16) | 7, (26 << 16) | 6, (27 << 16) | 6, (26 << 16) | 7, (27 << 16) | 7,
    (28 << 16) | 4, (29 << 16) | 4, (28 << 16) | 5, (29 << 16) | 5, (30 << 16) | 4, (31 << 16) | 4, (30 << 16) | 5, (31 << 16) | 5,
    (28 << 16) | 6, (29 << 16) | 6, (28 << 16) | 7, (29 << 16) | 7, (30 << 16) | 6, (31 << 16) | 6, (30 << 16) | 7, (31 << 16) | 7
};
