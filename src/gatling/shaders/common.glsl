#ifndef GATLING_SHADER_COMMON
#define GATLING_SHADER_COMMON

/* Unfortunately, 'vertex' as a name leads to issues with MoltenVK. */
struct face_vertex
{
    float pos_x;
    float pos_y;
    float pos_z;
    float norm_x;
    float norm_y;
    float norm_z;
    float u;
    float v;
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
    aabb left_aabb;
    uint left_child_offset;
    uint left_child_count;
    aabb right_aabb;
    uint right_child_offset;
    uint right_child_count;
};

struct path_segment
{
    vec3 pos;
    uint pixel_index;
    vec3 dir;
    uint rec_depth;
};

struct hit_info
{
    vec4  pos;
    float bc_u;
    float bc_v;
    uint  pixel_index;
    uint  face_index;
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
    face_vertex vertices[];
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
