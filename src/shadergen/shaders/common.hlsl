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
StructuredBuffer<fvertex> vertices;

uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float random_float_between_0_and_1(inout uint seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return float(seed) * (1.0 / 4294967296.0);
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
