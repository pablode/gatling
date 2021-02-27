const float FLOAT_MAX = 3.402823466e38;
const float FLOAT_MIN = 1.175494351e-38;
const float PI = 3.1415926535897932384626433832795;
const float TRI_EPS = 0.0000001;
const float RAY_OFFSET_EPS = 0.00001;

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
    uint mat_index;
};

struct material
{
    vec4 albedo;
    vec3 emission;
    float padding;
};

struct bvh_node
{
    /* Quantization frame. */
    vec3 p;                  /* 12 bytes */
    u8vec3 e;                /*  3 bytes */
    /* Indexing info. */
    uint8_t imask;           /*  1 byte  */
    uint child_index;        /*  4 bytes */
    uint face_index;         /*  4 bytes */
    uint meta[2];            /*  8 bytes */
    /* Child data. */
    u8vec4 q_lo_x[2];        /*  8 bytes */
    u8vec4 q_lo_y[2];        /*  8 bytes */
    u8vec4 q_lo_z[2];        /*  8 bytes */
    u8vec4 q_hi_x[2];        /*  8 bytes */
    u8vec4 q_hi_y[2];        /*  8 bytes */
    u8vec4 q_hi_z[2];        /*  8 bytes */
};

struct hit_info
{
    vec3 pos;
    uint face_index;
    vec2 bc;
    vec2 padding;
};

layout(set=0, binding=0) buffer BufferOutput
{
    vec4 pixels[];
};

layout(set=0, binding=1, std430) readonly buffer BufferBvhNodes
{
    bvh_node bvh_nodes[];
};

layout(set=0, binding=2) readonly buffer BufferFaces
{
    face faces[];
};

layout(set=0, binding=3) readonly buffer BufferVertices
{
    fvertex vertices[];
};

layout(set=0, binding=4) readonly buffer BufferMaterials
{
    material materials[];
};

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
