const float FLOAT_MAX = 3.402823466e38;
const float FLOAT_MIN = 1.175494351e-38;
const float PI = 3.1415926535897932384626433832795;
const float TRI_EPS = 0.000000001;

// These must match the IDs in gi.h
#define AOV_ID_COLOR 0
#define AOV_ID_NORMAL 1
#define AOV_ID_DEBUG_NEE 2
#define AOV_ID_DEBUG_BARYCENTRICS 3
#define AOV_ID_DEBUG_TEXCOORDS 4
#define AOV_ID_DEBUG_BOUNCES 5
#define AOV_ID_DEBUG_CLOCK_CYCLES 6

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
    uint mat_idx;
};

struct RayInfo
{
    vec3  origin;
    float tmax;
    vec3  dir;
    float padding;
};

// RNG producing on a four-element vector.
// From: "Hash Functions for GPU Rendering" by Jarzynski and Olano.
// Licensed under CC BY-ND 3.0: https://creativecommons.org/licenses/by-nd/3.0/
uvec4 pcg4d(uvec4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

uvec4 pcg4d_init(uvec2 pixel_coords, uint frame_num)
{
    return uvec4(pixel_coords.xy, frame_num, 0);
}

vec4 uvec4AsVec4(uvec4 v)
{
    v >>= 9;
    v |= 0x3f800000;
    return vec4(uintBitsToFloat(v)) - vec4(1.0, 1.0, 1.0, 1.0);
}

vec4 pcg4d_next(inout uvec4 rng_state)
{
    rng_state.w++;
    return uvec4AsVec4(pcg4d(rng_state));
}

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

#ifdef DYNAMIC_OFFSETTING
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
    const float EPS = 0.00001;
    return p + geom_normal * EPS;
}
#endif

// Hand-crafted Morton code lookup table; index corresponds to thread-index in group.
const uint16_t MORTON_2D_LUT_32x8[256] = {
    uint16_t(( 0 << 8) | 0), uint16_t(( 1 << 8) | 0), uint16_t(( 0 << 8) | 1), uint16_t(( 1 << 8) | 1),
    uint16_t(( 2 << 8) | 0), uint16_t(( 3 << 8) | 0), uint16_t(( 2 << 8) | 1), uint16_t(( 3 << 8) | 1),
    uint16_t(( 0 << 8) | 2), uint16_t(( 1 << 8) | 2), uint16_t(( 0 << 8) | 3), uint16_t(( 1 << 8) | 3),
    uint16_t(( 2 << 8) | 2), uint16_t(( 3 << 8) | 2), uint16_t(( 2 << 8) | 3), uint16_t(( 3 << 8) | 3),
    uint16_t(( 4 << 8) | 0), uint16_t(( 5 << 8) | 0), uint16_t(( 4 << 8) | 1), uint16_t(( 5 << 8) | 1),
    uint16_t(( 6 << 8) | 0), uint16_t(( 7 << 8) | 0), uint16_t(( 6 << 8) | 1), uint16_t(( 7 << 8) | 1),
    uint16_t(( 4 << 8) | 2), uint16_t(( 5 << 8) | 2), uint16_t(( 4 << 8) | 3), uint16_t(( 5 << 8) | 3),
    uint16_t(( 6 << 8) | 2), uint16_t(( 7 << 8) | 2), uint16_t(( 6 << 8) | 3), uint16_t(( 7 << 8) | 3),
    uint16_t(( 0 << 8) | 4), uint16_t(( 1 << 8) | 4), uint16_t(( 0 << 8) | 5), uint16_t(( 1 << 8) | 5),
    uint16_t(( 2 << 8) | 4), uint16_t(( 3 << 8) | 4), uint16_t(( 2 << 8) | 5), uint16_t(( 3 << 8) | 5),
    uint16_t(( 0 << 8) | 6), uint16_t(( 1 << 8) | 6), uint16_t(( 0 << 8) | 7), uint16_t(( 1 << 8) | 7),
    uint16_t(( 2 << 8) | 6), uint16_t(( 3 << 8) | 6), uint16_t(( 2 << 8) | 7), uint16_t(( 3 << 8) | 7),
    uint16_t(( 4 << 8) | 4), uint16_t(( 5 << 8) | 4), uint16_t(( 4 << 8) | 5), uint16_t(( 5 << 8) | 5),
    uint16_t(( 6 << 8) | 4), uint16_t(( 7 << 8) | 4), uint16_t(( 6 << 8) | 5), uint16_t(( 7 << 8) | 5),
    uint16_t(( 4 << 8) | 6), uint16_t(( 5 << 8) | 6), uint16_t(( 4 << 8) | 7), uint16_t(( 5 << 8) | 7),
    uint16_t(( 6 << 8) | 6), uint16_t(( 7 << 8) | 6), uint16_t(( 6 << 8) | 7), uint16_t(( 7 << 8) | 7),
    uint16_t(( 8 << 8) | 0), uint16_t(( 9 << 8) | 0), uint16_t(( 8 << 8) | 1), uint16_t(( 9 << 8) | 1),
    uint16_t((10 << 8) | 0), uint16_t((11 << 8) | 0), uint16_t((10 << 8) | 1), uint16_t((11 << 8) | 1),
    uint16_t(( 8 << 8) | 2), uint16_t(( 9 << 8) | 2), uint16_t(( 8 << 8) | 3), uint16_t(( 9 << 8) | 3),
    uint16_t((10 << 8) | 2), uint16_t((11 << 8) | 2), uint16_t((10 << 8) | 3), uint16_t((11 << 8) | 3),
    uint16_t((12 << 8) | 0), uint16_t((13 << 8) | 0), uint16_t((12 << 8) | 1), uint16_t((13 << 8) | 1),
    uint16_t((14 << 8) | 0), uint16_t((15 << 8) | 0), uint16_t((14 << 8) | 1), uint16_t((15 << 8) | 1),
    uint16_t((12 << 8) | 2), uint16_t((13 << 8) | 2), uint16_t((12 << 8) | 3), uint16_t((13 << 8) | 3),
    uint16_t((14 << 8) | 2), uint16_t((15 << 8) | 2), uint16_t((14 << 8) | 3), uint16_t((15 << 8) | 3),
    uint16_t(( 8 << 8) | 4), uint16_t(( 9 << 8) | 4), uint16_t(( 8 << 8) | 5), uint16_t(( 9 << 8) | 5),
    uint16_t((10 << 8) | 4), uint16_t((11 << 8) | 4), uint16_t((10 << 8) | 5), uint16_t((11 << 8) | 5),
    uint16_t(( 8 << 8) | 6), uint16_t(( 9 << 8) | 6), uint16_t(( 8 << 8) | 7), uint16_t(( 9 << 8) | 7),
    uint16_t((10 << 8) | 6), uint16_t((11 << 8) | 6), uint16_t((10 << 8) | 7), uint16_t((11 << 8) | 7),
    uint16_t((12 << 8) | 4), uint16_t((13 << 8) | 4), uint16_t((12 << 8) | 5), uint16_t((13 << 8) | 5),
    uint16_t((14 << 8) | 4), uint16_t((15 << 8) | 4), uint16_t((14 << 8) | 5), uint16_t((15 << 8) | 5),
    uint16_t((12 << 8) | 6), uint16_t((13 << 8) | 6), uint16_t((12 << 8) | 7), uint16_t((13 << 8) | 7),
    uint16_t((14 << 8) | 6), uint16_t((15 << 8) | 6), uint16_t((14 << 8) | 7), uint16_t((15 << 8) | 7),
    uint16_t((16 << 8) | 0), uint16_t((17 << 8) | 0), uint16_t((16 << 8) | 1), uint16_t((17 << 8) | 1),
    uint16_t((18 << 8) | 0), uint16_t((19 << 8) | 0), uint16_t((18 << 8) | 1), uint16_t((19 << 8) | 1),
    uint16_t((16 << 8) | 2), uint16_t((17 << 8) | 2), uint16_t((16 << 8) | 3), uint16_t((17 << 8) | 3),
    uint16_t((18 << 8) | 2), uint16_t((19 << 8) | 2), uint16_t((18 << 8) | 3), uint16_t((19 << 8) | 3),
    uint16_t((20 << 8) | 0), uint16_t((21 << 8) | 0), uint16_t((20 << 8) | 1), uint16_t((21 << 8) | 1),
    uint16_t((22 << 8) | 0), uint16_t((23 << 8) | 0), uint16_t((22 << 8) | 1), uint16_t((23 << 8) | 1),
    uint16_t((20 << 8) | 2), uint16_t((21 << 8) | 2), uint16_t((20 << 8) | 3), uint16_t((21 << 8) | 3),
    uint16_t((22 << 8) | 2), uint16_t((23 << 8) | 2), uint16_t((22 << 8) | 3), uint16_t((23 << 8) | 3),
    uint16_t((16 << 8) | 4), uint16_t((17 << 8) | 4), uint16_t((16 << 8) | 5), uint16_t((17 << 8) | 5),
    uint16_t((18 << 8) | 4), uint16_t((19 << 8) | 4), uint16_t((18 << 8) | 5), uint16_t((19 << 8) | 5),
    uint16_t((16 << 8) | 6), uint16_t((17 << 8) | 6), uint16_t((16 << 8) | 7), uint16_t((17 << 8) | 7),
    uint16_t((18 << 8) | 6), uint16_t((19 << 8) | 6), uint16_t((18 << 8) | 7), uint16_t((19 << 8) | 7),
    uint16_t((20 << 8) | 4), uint16_t((21 << 8) | 4), uint16_t((20 << 8) | 5), uint16_t((21 << 8) | 5),
    uint16_t((22 << 8) | 4), uint16_t((23 << 8) | 4), uint16_t((22 << 8) | 5), uint16_t((23 << 8) | 5),
    uint16_t((20 << 8) | 6), uint16_t((21 << 8) | 6), uint16_t((20 << 8) | 7), uint16_t((21 << 8) | 7),
    uint16_t((22 << 8) | 6), uint16_t((23 << 8) | 6), uint16_t((22 << 8) | 7), uint16_t((23 << 8) | 7),
    uint16_t((24 << 8) | 0), uint16_t((25 << 8) | 0), uint16_t((24 << 8) | 1), uint16_t((25 << 8) | 1),
    uint16_t((26 << 8) | 0), uint16_t((27 << 8) | 0), uint16_t((26 << 8) | 1), uint16_t((27 << 8) | 1),
    uint16_t((24 << 8) | 2), uint16_t((25 << 8) | 2), uint16_t((24 << 8) | 3), uint16_t((25 << 8) | 3),
    uint16_t((26 << 8) | 2), uint16_t((27 << 8) | 2), uint16_t((26 << 8) | 3), uint16_t((27 << 8) | 3),
    uint16_t((28 << 8) | 0), uint16_t((29 << 8) | 0), uint16_t((28 << 8) | 1), uint16_t((29 << 8) | 1),
    uint16_t((30 << 8) | 0), uint16_t((31 << 8) | 0), uint16_t((30 << 8) | 1), uint16_t((31 << 8) | 1),
    uint16_t((28 << 8) | 2), uint16_t((29 << 8) | 2), uint16_t((28 << 8) | 3), uint16_t((29 << 8) | 3),
    uint16_t((30 << 8) | 2), uint16_t((31 << 8) | 2), uint16_t((30 << 8) | 3), uint16_t((31 << 8) | 3),
    uint16_t((24 << 8) | 4), uint16_t((25 << 8) | 4), uint16_t((24 << 8) | 5), uint16_t((25 << 8) | 5),
    uint16_t((26 << 8) | 4), uint16_t((27 << 8) | 4), uint16_t((26 << 8) | 5), uint16_t((27 << 8) | 5),
    uint16_t((24 << 8) | 6), uint16_t((25 << 8) | 6), uint16_t((24 << 8) | 7), uint16_t((25 << 8) | 7),
    uint16_t((26 << 8) | 6), uint16_t((27 << 8) | 6), uint16_t((26 << 8) | 7), uint16_t((27 << 8) | 7),
    uint16_t((28 << 8) | 4), uint16_t((29 << 8) | 4), uint16_t((28 << 8) | 5), uint16_t((29 << 8) | 5),
    uint16_t((30 << 8) | 4), uint16_t((31 << 8) | 4), uint16_t((30 << 8) | 5), uint16_t((31 << 8) | 5),
    uint16_t((28 << 8) | 6), uint16_t((29 << 8) | 6), uint16_t((28 << 8) | 7), uint16_t((29 << 8) | 7),
    uint16_t((30 << 8) | 6), uint16_t((31 << 8) | 6), uint16_t((30 << 8) | 7), uint16_t((31 << 8) | 7)
};

const uint16_t MORTON_2D_LUT_32x8_REV[256] = {
    uint16_t(  0), uint16_t(  1), uint16_t(  4), uint16_t(  5), uint16_t( 16), uint16_t( 17), uint16_t( 20), uint16_t( 21),
    uint16_t( 64), uint16_t( 65), uint16_t( 68), uint16_t( 69), uint16_t( 80), uint16_t( 81), uint16_t( 84), uint16_t( 85),
    uint16_t(128), uint16_t(129), uint16_t(132), uint16_t(133), uint16_t(144), uint16_t(145), uint16_t(148), uint16_t(149),
    uint16_t(192), uint16_t(193), uint16_t(196), uint16_t(197), uint16_t(208), uint16_t(209), uint16_t(212), uint16_t(213),
    uint16_t(  2), uint16_t(  3), uint16_t(  6), uint16_t(  7), uint16_t( 18), uint16_t( 19), uint16_t( 22), uint16_t( 23),
    uint16_t( 66), uint16_t( 67), uint16_t( 70), uint16_t( 71), uint16_t( 82), uint16_t( 83), uint16_t( 86), uint16_t( 87),
    uint16_t(130), uint16_t(131), uint16_t(134), uint16_t(135), uint16_t(146), uint16_t(147), uint16_t(150), uint16_t(151),
    uint16_t(194), uint16_t(195), uint16_t(198), uint16_t(199), uint16_t(210), uint16_t(211), uint16_t(214), uint16_t(215),
    uint16_t(  8), uint16_t(  9), uint16_t( 12), uint16_t( 13), uint16_t( 24), uint16_t( 25), uint16_t( 28), uint16_t( 29),
    uint16_t( 72), uint16_t( 73), uint16_t( 76), uint16_t( 77), uint16_t( 88), uint16_t( 89), uint16_t( 92), uint16_t( 93),
    uint16_t(136), uint16_t(137), uint16_t(140), uint16_t(141), uint16_t(152), uint16_t(153), uint16_t(156), uint16_t(157),
    uint16_t(200), uint16_t(201), uint16_t(204), uint16_t(205), uint16_t(216), uint16_t(217), uint16_t(220), uint16_t(221),
    uint16_t( 10), uint16_t( 11), uint16_t( 14), uint16_t( 15), uint16_t( 26), uint16_t( 27), uint16_t( 30), uint16_t( 31),
    uint16_t( 74), uint16_t( 75), uint16_t( 78), uint16_t( 79), uint16_t( 90), uint16_t( 91), uint16_t( 94), uint16_t( 95),
    uint16_t(138), uint16_t(139), uint16_t(142), uint16_t(143), uint16_t(154), uint16_t(155), uint16_t(158), uint16_t(159),
    uint16_t(202), uint16_t(203), uint16_t(206), uint16_t(207), uint16_t(218), uint16_t(219), uint16_t(222), uint16_t(223),
    uint16_t( 32), uint16_t( 33), uint16_t( 36), uint16_t( 37), uint16_t( 48), uint16_t( 49), uint16_t( 52), uint16_t( 53),
    uint16_t( 96), uint16_t( 97), uint16_t(100), uint16_t(101), uint16_t(112), uint16_t(113), uint16_t(116), uint16_t(117),
    uint16_t(160), uint16_t(161), uint16_t(164), uint16_t(165), uint16_t(176), uint16_t(177), uint16_t(180), uint16_t(181),
    uint16_t(224), uint16_t(225), uint16_t(228), uint16_t(229), uint16_t(240), uint16_t(241), uint16_t(244), uint16_t(245),
    uint16_t( 34), uint16_t( 35), uint16_t( 38), uint16_t( 39), uint16_t( 50), uint16_t( 51), uint16_t( 54), uint16_t( 55),
    uint16_t( 98), uint16_t( 99), uint16_t(102), uint16_t(103), uint16_t(114), uint16_t(115), uint16_t(118), uint16_t(119),
    uint16_t(162), uint16_t(163), uint16_t(166), uint16_t(167), uint16_t(178), uint16_t(179), uint16_t(182), uint16_t(183),
    uint16_t(226), uint16_t(227), uint16_t(230), uint16_t(231), uint16_t(242), uint16_t(243), uint16_t(246), uint16_t(247),
    uint16_t( 40), uint16_t( 41), uint16_t( 44), uint16_t( 45), uint16_t( 56), uint16_t( 57), uint16_t( 60), uint16_t( 61),
    uint16_t(104), uint16_t(105), uint16_t(108), uint16_t(109), uint16_t(120), uint16_t(121), uint16_t(124), uint16_t(125),
    uint16_t(168), uint16_t(169), uint16_t(172), uint16_t(173), uint16_t(184), uint16_t(185), uint16_t(188), uint16_t(189),
    uint16_t(232), uint16_t(233), uint16_t(236), uint16_t(237), uint16_t(248), uint16_t(249), uint16_t(252), uint16_t(253),
    uint16_t( 42), uint16_t( 43), uint16_t( 46), uint16_t( 47), uint16_t( 58), uint16_t( 59), uint16_t( 62), uint16_t( 63),
    uint16_t(106), uint16_t(107), uint16_t(110), uint16_t(111), uint16_t(122), uint16_t(123), uint16_t(126), uint16_t(127),
    uint16_t(170), uint16_t(171), uint16_t(174), uint16_t(175), uint16_t(186), uint16_t(187), uint16_t(190), uint16_t(191),
    uint16_t(234), uint16_t(235), uint16_t(238), uint16_t(239), uint16_t(250), uint16_t(251), uint16_t(254), uint16_t(255)
};
