#ifndef GML_H
#define GML_H

#include <stdint.h>
#include <stdbool.h>

#define GML_MIN_FUNC(PREFIX, TYPE)  \
  TYPE PREFIX##min(TYPE a, TYPE b);
#define GML_MAX_FUNC(PREFIX, TYPE)  \
  TYPE PREFIX##max(TYPE a, TYPE b);

GML_MIN_FUNC(i, int)
GML_MIN_FUNC(i32, int32_t)
GML_MIN_FUNC(i64, int64_t)
GML_MIN_FUNC(u, unsigned int)
GML_MIN_FUNC(u32, uint32_t)
GML_MIN_FUNC(u64, uint64_t)

GML_MAX_FUNC(i, int)
GML_MAX_FUNC(i32, int32_t)
GML_MAX_FUNC(i64, int64_t)
GML_MAX_FUNC(u, unsigned int)
GML_MAX_FUNC(u32, uint32_t)
GML_MAX_FUNC(u64, uint64_t)

#undef GML_MIN_FUNC
#undef GML_MAX_FUNC

typedef float gml_vec3[3];

void gml_vec3_assign(const gml_vec3 a, gml_vec3 b);
void gml_vec3_add(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_sub(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_div(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_divs(const gml_vec3 a, float s, gml_vec3 b);
void gml_vec3_mul(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_muls(const gml_vec3 a, float s, gml_vec3 b);
float gml_vec3_dot(const gml_vec3 a, const gml_vec3 b);
void gml_vec3_cross(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_lerp(const gml_vec3 a, const gml_vec3 b, float t, gml_vec3 v);
float gml_vec3_length(const gml_vec3 v);
void gml_vec3_normalize(const gml_vec3 a, gml_vec3 b);
void gml_vec3_max(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_min(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
float gml_vec3_mincomp(const gml_vec3 v);
float gml_vec3_maxcomp(const gml_vec3 v);

typedef float gml_vec4[4];

void gml_vec4_assign(const gml_vec4 a, gml_vec4 b);

typedef float gml_mat4[4][4];

void gml_mat4_assign(const gml_mat4 a, gml_mat4 b);
void gml_mat4_mul(const gml_mat4 a, const gml_mat4 b, gml_mat4 c);
void gml_mat4_mul_vec4(const gml_mat4 a, const gml_vec4 b, gml_vec4 c);
void gml_mat4_identity(gml_mat4 a);

typedef float gml_mat3[3][3];

void gml_mat3_assign(const gml_mat3 a, gml_mat3 b);
void gml_mat3_from_mat4(const gml_mat4 a, gml_mat3 b);
bool gml_mat3_invert(const gml_mat3 a, gml_mat3 b);
void gml_mat3_transpose(const gml_mat3 a, gml_mat3 b);
void gml_mat3_mul_vec3(const gml_mat3 a, const gml_vec3 b, gml_vec3 c);

typedef struct gml_aabb {
  gml_vec3 min;
  gml_vec3 max;
} gml_aabb;

void gml_aabb_make_smallest(gml_aabb* aabb);
void gml_aabb_make_biggest(gml_aabb* aabb);
void gml_aabb_make_from_triangle(const gml_vec3 v_a, const gml_vec3 v_b, const gml_vec3 v_c, gml_aabb* aabb);
void gml_aabb_merge(const gml_aabb* aabb_a, const gml_aabb* aabb_b, gml_aabb* aabb_c);
void gml_aabb_include(const gml_aabb* aabb_a, const gml_vec3 v, gml_aabb* aabb_b);
void gml_aabb_intersect(const gml_aabb* aabb_a, const gml_aabb* aabb_b, gml_aabb* aabb_c);
void gml_aabb_size(const gml_aabb* aabb, gml_vec3 size);
float gml_aabb_half_area(const gml_aabb* aabb);
float gml_aabb_area(const gml_aabb* aabb);

#endif
