#ifndef GP_AABB_H
#define GP_AABB_H

#include <gml.h>

typedef struct gp_aabb {
  gp_vec3 min;
  gp_vec3 max;
} gp_aabb;

void gp_aabb_make_smallest(gp_aabb* aabb);
void gp_aabb_make_biggest(gp_aabb* aabb);
void gp_aabb_make_from_triangle(const gp_vec3 v_a, const gp_vec3 v_b, const gp_vec3 v_c, gp_aabb* aabb);
void gp_aabb_merge(const gp_aabb* aabb_a, const gp_aabb* aabb_b, gp_aabb* aabb_c);
void gp_aabb_include(const gp_aabb* aabb_a, const gp_vec3 v, gp_aabb* aabb_b);
void gp_aabb_intersect(const gp_aabb* aabb_a, const gp_aabb* aabb_b, gp_aabb* aabb_c);
void gp_aabb_size(const gp_aabb* aabb, gp_vec3 size);
float gp_aabb_half_area(const gp_aabb* aabb);
float gp_aabb_area(const gp_aabb* aabb);

#endif
