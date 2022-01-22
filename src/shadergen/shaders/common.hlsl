const static float FLOAT_MAX = 3.402823466e38;
const static float FLOAT_MIN = 1.175494351e-38;
const static float PI = 3.1415926535897932384626433832795;
const static float TRI_EPS = 0.000000001;

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

struct bvh_node
{
    uint4 f1;
    uint4 f2;
    uint4 f3;
    uint4 f4;
    uint4 f5;
};

struct RayInfo
{
  float3 origin;
  float  tmin;
  float3 dir;
  float  tmax;
};

struct Hit_info
{
    float3 pos;
    uint face_idx;
    float2 bc;
};

[[vk::binding(0)]]
RWStructuredBuffer<float4> pixels;

[[vk::binding(1)]]
StructuredBuffer<bvh_node> bvh_nodes;

[[vk::binding(2)]]
StructuredBuffer<face> faces;

[[vk::binding(3)]]
StructuredBuffer<uint> emissive_face_indices;

[[vk::binding(4)]]
StructuredBuffer<fvertex> vertices;

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
float3 offset_ray_origin(float3 p, float3 n)
{
    const float ORIGIN = 1.0 / 32.0;
    const float FLOAT_SCALE = 1.0 / 65536.0;
    const float INT_SCALE = 256.0;

    int3 of_i = int3(INT_SCALE * n.x, INT_SCALE * n.y, INT_SCALE * n.z);

    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0.0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0.0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0.0) ? -of_i.z : of_i.z))
    );

    return float3(
        abs(p.x) < ORIGIN ? (p.x + FLOAT_SCALE * n.x) : p_i.x,
        abs(p.y) < ORIGIN ? (p.y + FLOAT_SCALE * n.y) : p_i.y,
        abs(p.z) < ORIGIN ? (p.z + FLOAT_SCALE * n.z) : p_i.z
    );
}
