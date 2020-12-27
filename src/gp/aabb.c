#include "aabb.h"

#include <math.h>

void gp_aabb_make_smallest(gp_aabb* aabb)
{
  aabb->min[0] = +INFINITY;
  aabb->min[1] = +INFINITY;
  aabb->min[2] = +INFINITY;
  aabb->max[0] = -INFINITY;
  aabb->max[1] = -INFINITY;
  aabb->max[2] = -INFINITY;
}

void gp_aabb_make_biggest(gp_aabb* aabb)
{
  aabb->min[0] = -INFINITY;
  aabb->min[1] = -INFINITY;
  aabb->min[2] = -INFINITY;
  aabb->max[0] = +INFINITY;
  aabb->max[1] = +INFINITY;
  aabb->max[2] = +INFINITY;
}

void gp_aabb_make_from_triangle(const gml_vec3 v_a,
                                const gml_vec3 v_b,
                                const gml_vec3 v_c,
                                gp_aabb* aabb)
{
  aabb->min[0] = fminf(fminf(v_a[0], v_b[0]), v_c[0]);
  aabb->min[1] = fminf(fminf(v_a[1], v_b[1]), v_c[1]);
  aabb->min[2] = fminf(fminf(v_a[2], v_b[2]), v_c[2]);
  aabb->max[0] = fmaxf(fmaxf(v_a[0], v_b[0]), v_c[0]);
  aabb->max[1] = fmaxf(fmaxf(v_a[1], v_b[1]), v_c[1]);
  aabb->max[2] = fmaxf(fmaxf(v_a[2], v_b[2]), v_c[2]);
}

void gp_aabb_merge(const gp_aabb* aabb_a, const gp_aabb* aabb_b, gp_aabb* aabb_c)
{
  aabb_c->min[0] = fminf(aabb_a->min[0], aabb_b->min[0]);
  aabb_c->min[1] = fminf(aabb_a->min[1], aabb_b->min[1]);
  aabb_c->min[2] = fminf(aabb_a->min[2], aabb_b->min[2]);
  aabb_c->max[0] = fmaxf(aabb_a->max[0], aabb_b->max[0]);
  aabb_c->max[1] = fmaxf(aabb_a->max[1], aabb_b->max[1]);
  aabb_c->max[2] = fmaxf(aabb_a->max[2], aabb_b->max[2]);
}

void gp_aabb_include(const gp_aabb* aabb_a, const gml_vec3 v, gp_aabb* aabb_b)
{
  aabb_b->min[0] = fminf(aabb_a->min[0], v[0]);
  aabb_b->min[1] = fminf(aabb_a->min[1], v[1]);
  aabb_b->min[2] = fminf(aabb_a->min[2], v[2]);
  aabb_b->max[0] = fmaxf(aabb_a->max[0], v[0]);
  aabb_b->max[1] = fmaxf(aabb_a->max[1], v[1]);
  aabb_b->max[2] = fmaxf(aabb_a->max[2], v[2]);
}

void gp_aabb_intersect(
  const gp_aabb* aabb_a,
  const gp_aabb* aabb_b,
  gp_aabb* aabb_c)
{
  aabb_c->min[0] = fmaxf(aabb_a->min[0], aabb_b->min[0]);
  aabb_c->min[1] = fmaxf(aabb_a->min[1], aabb_b->min[1]);
  aabb_c->min[2] = fmaxf(aabb_a->min[2], aabb_b->min[2]);
  aabb_c->max[0] = fminf(aabb_a->max[0], aabb_b->max[0]);
  aabb_c->max[1] = fminf(aabb_a->max[1], aabb_b->max[1]);
  aabb_c->max[2] = fminf(aabb_a->max[2], aabb_b->max[2]);
}

void gp_aabb_size(const gp_aabb* aabb, gml_vec3 size)
{
  size[0] = fmaxf(0.0f, aabb->max[0] - aabb->min[0]);
  size[1] = fmaxf(0.0f, aabb->max[1] - aabb->min[1]);
  size[2] = fmaxf(0.0f, aabb->max[2] - aabb->min[2]);
}

float gp_aabb_half_area(const gp_aabb* aabb)
{
  gml_vec3 size;
  gp_aabb_size(aabb, size);
  return size[0] * size[1] + size[0] * size[2] + size[1] * size[2];
}

float gp_aabb_area(const gp_aabb* aabb)
{
  return 2.0f * gp_aabb_half_area(aabb);
}
