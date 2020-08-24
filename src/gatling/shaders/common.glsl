#ifndef GATLING_SHADER_COMMON
#define GATLING_SHADER_COMMON

#define FLOAT_MAX 3.402823466e38
#define FLOAT_MIN 1.175494351e-38f

struct vertex
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
    vec4 color;
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

struct path_segment
{
    vec3 origin;
    uint pixel_index;
    vec3 dir;
    uint rec_depth;
};

struct hit_info
{
    vec4 pos;
    vec2 bc;
    uint pixel_index;
    uint face_index;
};

layout(set=0, binding=0) queuefamilycoherent buffer BufferOutput
{
    uint pixels[];
};

layout(set=0, binding=1) buffer BufferPathSegments
{
    uint path_segment_counter;
    uint pad_0;
    uint pad_1;
    uint pad_2;
    path_segment path_segments[];
};

layout(set=0, binding=2) readonly buffer BufferBvhNodes
{
    bvh_node bvh_nodes[];
};

layout(set=0, binding=3) readonly buffer BufferFaces
{
    face faces[];
};

layout(set=0, binding=4) readonly buffer BufferVertices
{
    vertex vertices[];
};

layout(set=0, binding=5) readonly buffer BufferMaterials
{
    material materials[];
};

layout(set=0, binding=6) buffer BufferHitInfos
{
    uint hit_write_counter;
    uint hit_read_counter;
    uint padding[2];
    hit_info hits[];
};

#endif
