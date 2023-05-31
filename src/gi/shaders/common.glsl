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

// Octahedral encoding as described in Listings 1, 2 of Cigolle et al.
// https://jcgt.org/published/0003/02/01/paper.pdf
vec2 signNonZero(vec2 v)
{
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

vec2 encode_octahedral(vec3 v)
{
    v /= (abs(v.x) + abs(v.y) + abs(v.z));
    return (v.z < 0.0) ? ((1.0 - abs(v.yx)) * signNonZero(v.xy)) : v.xy;
}

vec3 decode_octahedral(vec2 e)
{
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    // Optimization by Rune Stubbe: https://twitter.com/Stubbesaurus/status/937994790553227264
    float t = max(-v.z, 0.0);
    v.x += v.x >= 0.0 ? -t : t;
    v.y += v.y >= 0.0 ? -t : t;
    return normalize(v);
}

#endif
