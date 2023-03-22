#ifndef H_COMMON
#define H_COMMON

const uint UINT32_MAX = 0xFFFFFFFFu;
const float FLOAT_MAX = 3.402823466e38;
const float FLOAT_MIN = 1.175494351e-38;
const float PI = 3.1415926535897932384626433832795;

#ifndef NDEBUG
#extension GL_EXT_debug_printf: enable
#define ASSERT(cond, str) if (!(cond)) debugPrintfEXT(str)
#else
#define ASSERT(cond, str)
#endif

// These must match the IDs in gi.h
#define AOV_ID_COLOR 0
#define AOV_ID_NORMAL 1
#define AOV_ID_DEBUG_NEE 2
#define AOV_ID_DEBUG_BARYCENTRICS 3
#define AOV_ID_DEBUG_TEXCOORDS 4
#define AOV_ID_DEBUG_BOUNCES 5
#define AOV_ID_DEBUG_CLOCK_CYCLES 6
#define AOV_ID_DEBUG_OPACITY 7

struct fvertex
{
    /* pos.{x, y, z}, tex.u */
    vec4 field1;
    /* norm.{x, y, z}, tex.v */
    vec4 field2;
};

struct face
{
    uint v_0;
    uint v_1;
    uint v_2;
};

vec4 uvec4AsVec4(uvec4 v)
{
    v >>= 9;
    v |= 0x3f800000;
    return vec4(uintBitsToFloat(v)) - vec4(1.0);
}

float uintAsFloat(uint v)
{
    return uintBitsToFloat(0x3f800000 | (v >> 9)) - 1.0;
}

// Enable for higher quality random numbers at the cost of performance
#define RAND_4D

#ifdef RAND_4D
// RNG producing on a four-element vector.
// From: "Hash Functions for GPU Rendering" by Jarzynski and Olano.
// Licensed under CC BY-ND 3.0: https://creativecommons.org/licenses/by-nd/3.0/
uvec4 hash_pcg4d(uvec4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

vec4 rng4d_next(inout uvec4 rng_state)
{
    rng_state.w++;
    return uvec4AsVec4(hash_pcg4d(rng_state));
}

uvec4 rng4d_init(uvec2 pixel_coords, uint frame_num)
{
    return uvec4(pixel_coords.xy, frame_num, 0);
}
#else
// Hash prospector parametrization found by GH user TheIronBorn:
// https://github.com/skeeto/hash-prospector#discovered-hash-functions
uint hash_theironborn(uint x)
{
    x ^= x >> 16u;
    x *= 0x21f0aaadu;
    x ^= x >> 15u;
    x *= 0xd35a2d97u;
    x ^= x >> 15u;
    return x;
}

float rng_next(inout uint rng_state)
{
    rng_state = hash_theironborn(rng_state);
    return uintAsFloat(rng_state);
}

uint rng_init(uint pixel_index, uint frame_num)
{
    return pixel_index ^ hash_theironborn(frame_num);
}
#endif

// Duff et al. 2017. Building an Orthonormal Basis, Revisited. JCGT.
// Licensed under CC BY-ND 3.0: https://creativecommons.org/licenses/by-nd/3.0/
void orthonormal_basis(in vec3 n, out vec3 b1, out vec3 b2)
{
    float nsign = (n.z >= 0.0 ? 1.0 : -1.0); // sign() intrinsic returns 0.0 for 0.0 :(
    float a = -1.0 / (nsign + n.z);
    float b = n.x * n.y * a;

    b1 = vec3(1.0 + nsign * n.x * n.x * a, nsign * b, -nsign * n.x);
    b2 = vec3(b, nsign + n.y * n.y * a, -n.y);
}

#if 1
// From: "A Fast and Robust Method for Avoiding Self-Intersection"
// Wächter and Binder, Ch. 6 in Ray Tracing Gems I.
// Licensed under CC BY-NC-ND 4.0: https://creativecommons.org/licenses/by-nc-nd/4.0/
vec3 offset_ray_origin(vec3 p, vec3 geom_normal)
{
    const float origin = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale = 256.0;

    ivec3 of_i = ivec3(
        int_scale * geom_normal.x,
        int_scale * geom_normal.y,
        int_scale * geom_normal.z
    );

    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0.0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0.0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0.0) ? -of_i.z : of_i.z))
    );

    return vec3(
        abs(p.x) < origin ? (p.x + float_scale * geom_normal.x) : p_i.x,
        abs(p.y) < origin ? (p.y + float_scale * geom_normal.y) : p_i.y,
        abs(p.z) < origin ? (p.z + float_scale * geom_normal.z) : p_i.z
    );
}
#else
vec3 offset_ray_origin(vec3 p, vec3 geom_normal)
{
    const float EPS = 0.0001;
    return p + geom_normal * EPS;
}
#endif

// Hand-crafted Morton code lookup table; index corresponds to thread-index in group.
const uint MORTON_2D_LUT_32x8[256] = {
    (( 0 << 8) | 0), (( 1 << 8) | 0), (( 0 << 8) | 1), (( 1 << 8) | 1),
    (( 2 << 8) | 0), (( 3 << 8) | 0), (( 2 << 8) | 1), (( 3 << 8) | 1),
    (( 0 << 8) | 2), (( 1 << 8) | 2), (( 0 << 8) | 3), (( 1 << 8) | 3),
    (( 2 << 8) | 2), (( 3 << 8) | 2), (( 2 << 8) | 3), (( 3 << 8) | 3),
    (( 4 << 8) | 0), (( 5 << 8) | 0), (( 4 << 8) | 1), (( 5 << 8) | 1),
    (( 6 << 8) | 0), (( 7 << 8) | 0), (( 6 << 8) | 1), (( 7 << 8) | 1),
    (( 4 << 8) | 2), (( 5 << 8) | 2), (( 4 << 8) | 3), (( 5 << 8) | 3),
    (( 6 << 8) | 2), (( 7 << 8) | 2), (( 6 << 8) | 3), (( 7 << 8) | 3),
    (( 0 << 8) | 4), (( 1 << 8) | 4), (( 0 << 8) | 5), (( 1 << 8) | 5),
    (( 2 << 8) | 4), (( 3 << 8) | 4), (( 2 << 8) | 5), (( 3 << 8) | 5),
    (( 0 << 8) | 6), (( 1 << 8) | 6), (( 0 << 8) | 7), (( 1 << 8) | 7),
    (( 2 << 8) | 6), (( 3 << 8) | 6), (( 2 << 8) | 7), (( 3 << 8) | 7),
    (( 4 << 8) | 4), (( 5 << 8) | 4), (( 4 << 8) | 5), (( 5 << 8) | 5),
    (( 6 << 8) | 4), (( 7 << 8) | 4), (( 6 << 8) | 5), (( 7 << 8) | 5),
    (( 4 << 8) | 6), (( 5 << 8) | 6), (( 4 << 8) | 7), (( 5 << 8) | 7),
    (( 6 << 8) | 6), (( 7 << 8) | 6), (( 6 << 8) | 7), (( 7 << 8) | 7),
    (( 8 << 8) | 0), (( 9 << 8) | 0), (( 8 << 8) | 1), (( 9 << 8) | 1),
    ((10 << 8) | 0), ((11 << 8) | 0), ((10 << 8) | 1), ((11 << 8) | 1),
    (( 8 << 8) | 2), (( 9 << 8) | 2), (( 8 << 8) | 3), (( 9 << 8) | 3),
    ((10 << 8) | 2), ((11 << 8) | 2), ((10 << 8) | 3), ((11 << 8) | 3),
    ((12 << 8) | 0), ((13 << 8) | 0), ((12 << 8) | 1), ((13 << 8) | 1),
    ((14 << 8) | 0), ((15 << 8) | 0), ((14 << 8) | 1), ((15 << 8) | 1),
    ((12 << 8) | 2), ((13 << 8) | 2), ((12 << 8) | 3), ((13 << 8) | 3),
    ((14 << 8) | 2), ((15 << 8) | 2), ((14 << 8) | 3), ((15 << 8) | 3),
    (( 8 << 8) | 4), (( 9 << 8) | 4), (( 8 << 8) | 5), (( 9 << 8) | 5),
    ((10 << 8) | 4), ((11 << 8) | 4), ((10 << 8) | 5), ((11 << 8) | 5),
    (( 8 << 8) | 6), (( 9 << 8) | 6), (( 8 << 8) | 7), (( 9 << 8) | 7),
    ((10 << 8) | 6), ((11 << 8) | 6), ((10 << 8) | 7), ((11 << 8) | 7),
    ((12 << 8) | 4), ((13 << 8) | 4), ((12 << 8) | 5), ((13 << 8) | 5),
    ((14 << 8) | 4), ((15 << 8) | 4), ((14 << 8) | 5), ((15 << 8) | 5),
    ((12 << 8) | 6), ((13 << 8) | 6), ((12 << 8) | 7), ((13 << 8) | 7),
    ((14 << 8) | 6), ((15 << 8) | 6), ((14 << 8) | 7), ((15 << 8) | 7),
    ((16 << 8) | 0), ((17 << 8) | 0), ((16 << 8) | 1), ((17 << 8) | 1),
    ((18 << 8) | 0), ((19 << 8) | 0), ((18 << 8) | 1), ((19 << 8) | 1),
    ((16 << 8) | 2), ((17 << 8) | 2), ((16 << 8) | 3), ((17 << 8) | 3),
    ((18 << 8) | 2), ((19 << 8) | 2), ((18 << 8) | 3), ((19 << 8) | 3),
    ((20 << 8) | 0), ((21 << 8) | 0), ((20 << 8) | 1), ((21 << 8) | 1),
    ((22 << 8) | 0), ((23 << 8) | 0), ((22 << 8) | 1), ((23 << 8) | 1),
    ((20 << 8) | 2), ((21 << 8) | 2), ((20 << 8) | 3), ((21 << 8) | 3),
    ((22 << 8) | 2), ((23 << 8) | 2), ((22 << 8) | 3), ((23 << 8) | 3),
    ((16 << 8) | 4), ((17 << 8) | 4), ((16 << 8) | 5), ((17 << 8) | 5),
    ((18 << 8) | 4), ((19 << 8) | 4), ((18 << 8) | 5), ((19 << 8) | 5),
    ((16 << 8) | 6), ((17 << 8) | 6), ((16 << 8) | 7), ((17 << 8) | 7),
    ((18 << 8) | 6), ((19 << 8) | 6), ((18 << 8) | 7), ((19 << 8) | 7),
    ((20 << 8) | 4), ((21 << 8) | 4), ((20 << 8) | 5), ((21 << 8) | 5),
    ((22 << 8) | 4), ((23 << 8) | 4), ((22 << 8) | 5), ((23 << 8) | 5),
    ((20 << 8) | 6), ((21 << 8) | 6), ((20 << 8) | 7), ((21 << 8) | 7),
    ((22 << 8) | 6), ((23 << 8) | 6), ((22 << 8) | 7), ((23 << 8) | 7),
    ((24 << 8) | 0), ((25 << 8) | 0), ((24 << 8) | 1), ((25 << 8) | 1),
    ((26 << 8) | 0), ((27 << 8) | 0), ((26 << 8) | 1), ((27 << 8) | 1),
    ((24 << 8) | 2), ((25 << 8) | 2), ((24 << 8) | 3), ((25 << 8) | 3),
    ((26 << 8) | 2), ((27 << 8) | 2), ((26 << 8) | 3), ((27 << 8) | 3),
    ((28 << 8) | 0), ((29 << 8) | 0), ((28 << 8) | 1), ((29 << 8) | 1),
    ((30 << 8) | 0), ((31 << 8) | 0), ((30 << 8) | 1), ((31 << 8) | 1),
    ((28 << 8) | 2), ((29 << 8) | 2), ((28 << 8) | 3), ((29 << 8) | 3),
    ((30 << 8) | 2), ((31 << 8) | 2), ((30 << 8) | 3), ((31 << 8) | 3),
    ((24 << 8) | 4), ((25 << 8) | 4), ((24 << 8) | 5), ((25 << 8) | 5),
    ((26 << 8) | 4), ((27 << 8) | 4), ((26 << 8) | 5), ((27 << 8) | 5),
    ((24 << 8) | 6), ((25 << 8) | 6), ((24 << 8) | 7), ((25 << 8) | 7),
    ((26 << 8) | 6), ((27 << 8) | 6), ((26 << 8) | 7), ((27 << 8) | 7),
    ((28 << 8) | 4), ((29 << 8) | 4), ((28 << 8) | 5), ((29 << 8) | 5),
    ((30 << 8) | 4), ((31 << 8) | 4), ((30 << 8) | 5), ((31 << 8) | 5),
    ((28 << 8) | 6), ((29 << 8) | 6), ((28 << 8) | 7), ((29 << 8) | 7),
    ((30 << 8) | 6), ((31 << 8) | 6), ((30 << 8) | 7), ((31 << 8) | 7)
};

const uint MORTON_2D_LUT_32x8_REV[256] = {
    (  0), (  1), (  4), (  5), ( 16), ( 17), ( 20), ( 21),
    ( 64), ( 65), ( 68), ( 69), ( 80), ( 81), ( 84), ( 85),
    (128), (129), (132), (133), (144), (145), (148), (149),
    (192), (193), (196), (197), (208), (209), (212), (213),
    (  2), (  3), (  6), (  7), ( 18), ( 19), ( 22), ( 23),
    ( 66), ( 67), ( 70), ( 71), ( 82), ( 83), ( 86), ( 87),
    (130), (131), (134), (135), (146), (147), (150), (151),
    (194), (195), (198), (199), (210), (211), (214), (215),
    (  8), (  9), ( 12), ( 13), ( 24), ( 25), ( 28), ( 29),
    ( 72), ( 73), ( 76), ( 77), ( 88), ( 89), ( 92), ( 93),
    (136), (137), (140), (141), (152), (153), (156), (157),
    (200), (201), (204), (205), (216), (217), (220), (221),
    ( 10), ( 11), ( 14), ( 15), ( 26), ( 27), ( 30), ( 31),
    ( 74), ( 75), ( 78), ( 79), ( 90), ( 91), ( 94), ( 95),
    (138), (139), (142), (143), (154), (155), (158), (159),
    (202), (203), (206), (207), (218), (219), (222), (223),
    ( 32), ( 33), ( 36), ( 37), ( 48), ( 49), ( 52), ( 53),
    ( 96), ( 97), (100), (101), (112), (113), (116), (117),
    (160), (161), (164), (165), (176), (177), (180), (181),
    (224), (225), (228), (229), (240), (241), (244), (245),
    ( 34), ( 35), ( 38), ( 39), ( 50), ( 51), ( 54), ( 55),
    ( 98), ( 99), (102), (103), (114), (115), (118), (119),
    (162), (163), (166), (167), (178), (179), (182), (183),
    (226), (227), (230), (231), (242), (243), (246), (247),
    ( 40), ( 41), ( 44), ( 45), ( 56), ( 57), ( 60), ( 61),
    (104), (105), (108), (109), (120), (121), (124), (125),
    (168), (169), (172), (173), (184), (185), (188), (189),
    (232), (233), (236), (237), (248), (249), (252), (253),
    ( 42), ( 43), ( 46), ( 47), ( 58), ( 59), ( 62), ( 63),
    (106), (107), (110), (111), (122), (123), (126), (127),
    (170), (171), (174), (175), (186), (187), (190), (191),
    (234), (235), (238), (239), (250), (251), (254), (255)
};

#endif
