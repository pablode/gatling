#ifndef GATLING_SHADER_COMMON
#define GATLING_SHADER_COMMON

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
};

struct aabb
{
    float min_x;
    float min_y;
    float min_z;
    float max_x;
    float max_y;
    float max_z;
};

struct bvh_node
{
    aabb aabb;
    uint field1;
    uint field2;
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

layout(set=0, binding=0) writeonly buffer BufferOutput
{
    vec4 pixels[];
};

layout(set=0, binding=1) buffer BufferPathSegments
{
    uint path_segment_counter;
    uint pad_0;
    uint pad_1;
    uint pad_2;
    path_segment path_segments[];
};

layout(set=0, binding=2) readonly buffer BufferHeader
{
    uint node_offset;
    uint node_count;
    uint face_offset;
    uint face_count;
    uint vertex_offset;
    uint vertex_count;
    uint material_offset;
    uint material_count;
    aabb root_aabb;
};

layout(set=0, binding=3) readonly buffer BufferBvhNodes
{
    bvh_node bvh_nodes[];
};

layout(set=0, binding=4) readonly buffer BufferFaces
{
    face faces[];
};

layout(set=0, binding=5) readonly buffer BufferVertices
{
    vertex vertices[];
};

layout(set=0, binding=6) readonly buffer BufferMaterials
{
    material materials[];
};

layout(set=0, binding=7) buffer BufferHitInfos
{
    uint hit_write_counter;
    uint hit_read_counter;
    uint padding[2];
    hit_info hits[];
};

#endif