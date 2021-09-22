const static float FLOAT_MAX = 3.402823466e38;
const static float FLOAT_MIN = 1.175494351e-38;
const static float PI = 3.1415926535897932384626433832795;
const static float TRI_EPS = 0.0000001;
const static float RAY_OFFSET_EPS = 0.0001;

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
    uint mat_index;
};

struct material
{
    float4 albedo;
    float3 emission;
    float padding;
};

struct bvh_node
{
    uint4 f1;
    uint4 f2;
    uint4 f3;
    uint4 f4;
    uint4 f5;
};

struct hit_info
{
    float3 pos;
    uint face_index;
    float2 bc;
    float2 padding;
};

[[vk::binding(0)]]
RWStructuredBuffer<float4> pixels;

[[vk::binding(1)]]
StructuredBuffer<bvh_node> bvh_nodes;

[[vk::binding(2)]]
StructuredBuffer<face> faces;

[[vk::binding(3)]]
StructuredBuffer<fvertex> vertices;

[[vk::binding(4)]]
StructuredBuffer<material> materials;

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
