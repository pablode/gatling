#ifndef GP_MATH_H
#define GP_MATH_H

#include "gp.h"

#include <stdint.h>

typedef float gp_vec3[3];

typedef float gp_vec2[2];

typedef struct gp_aabb {
  gp_vec3 min;
  gp_vec3 max;
} gp_aabb;

typedef struct gp_triangle {
  gp_vec3 v[3];
} gp_triangle;

GP_INLINE void gp_vec3_add(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
GP_INLINE void gp_vec3_sub(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
GP_INLINE void gp_vec3_div(const gp_vec3 a, float s, gp_vec3 b);
GP_INLINE void gp_vec3_mul(const gp_vec3 a, float s, gp_vec3 b);
GP_INLINE float gp_vec3_dot(const gp_vec3 a, const gp_vec3 b);
GP_INLINE void gp_vec3_cross(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
GP_INLINE void gp_vec3_lerp(const gp_vec3 a, const gp_vec3 b, float t, gp_vec3 v);
GP_INLINE float gp_vec3_length(const gp_vec3 v);
GP_INLINE float gp_vec3_comp_min(const gp_vec3 v);
GP_INLINE float gp_vec3_comp_max(const gp_vec3 v);

void gp_aabb_make_smallest(gp_aabb* aabb);
void gp_aabb_make_biggest(gp_aabb* aabb);
void gp_aabb_make_from_triangle(const gp_vec3 v_a, const gp_vec3 v_b, const gp_vec3 v_c, gp_aabb* aabb);
void gp_aabb_merge(const gp_aabb* aabb_a, const gp_aabb* aabb_b, gp_aabb* aabb_c);
void gp_aabb_intersect(const gp_aabb* aabb_a, const gp_aabb* aabb_b, gp_aabb* aabb_c);
GP_INLINE void gp_aabb_size(const gp_aabb* aabb, gp_vec3 size);
float gp_aabb_half_area(const gp_aabb* aabb);
float gp_aabb_area(const gp_aabb* aabb);

#endif
