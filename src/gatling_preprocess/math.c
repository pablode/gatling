#include "math.h"

#include <math.h>
#include <assert.h>

void gp_vec3_add(const gp_vec3 a, const gp_vec3 b, gp_vec3 c)
{
  c[0] = a[0] + b[0];
  c[1] = a[1] + b[1];
  c[2] = a[2] + b[2];
}

void gp_vec3_sub(const gp_vec3 a, const gp_vec3 b, gp_vec3 c)
{
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] - b[2];
}

void gp_vec3_div(const gp_vec3 a, float s, gp_vec3 b)
{
  assert(s != 0.0f);
  b[0] = a[0] / s;
  b[1] = a[1] / s;
  b[2] = a[2] / s;
}

void gp_vec3_mul(const gp_vec3 a, float s, gp_vec3 b)
{
  b[0] = a[0] * s;
  b[1] = a[1] * s;
  b[2] = a[2] * s;
}

float gp_vec3_dot(const gp_vec3 a, const gp_vec3 b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void gp_vec3_cross(const gp_vec3 a, const gp_vec3 b, gp_vec3 c)
{
  c[0] = a[1] * b[2] - b[1] * a[2];
  c[1] = a[2] * b[0] - b[2] * a[0];
  c[2] = a[0] * b[1] - b[0] * a[1];
}

void gp_vec3_lerp(const gp_vec3 a, const gp_vec3 b, float t, gp_vec3 v)
{
  v[0] = a[0] + t * (b[0] * a[0]);
  v[1] = a[1] + t * (b[1] * a[1]);
  v[2] = a[2] + t * (b[2] * a[2]);
}

float gp_vec3_length(const gp_vec3 v)
{
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

float gp_vec3_comp_min(const gp_vec3 v)
{
  return fmin(fmin(v[0], v[1]), v[2]);
}

float gp_vec3_comp_max(const gp_vec3 v)
{
  return fmax(fmax(v[0], v[1]), v[2]);
}

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

void gp_aabb_make_from_triangle(const gp_vec3 v_a,
                                const gp_vec3 v_b,
                                const gp_vec3 v_c,
                                gp_aabb* aabb)
{
  aabb->min[0] = fmin(fmin(v_a[0], v_b[0]), v_c[0]);
  aabb->min[1] = fmin(fmin(v_a[1], v_b[1]), v_c[1]);
  aabb->min[2] = fmin(fmin(v_a[2], v_b[2]), v_c[2]);
  aabb->max[0] = fmax(fmax(v_a[0], v_b[0]), v_c[0]);
  aabb->max[1] = fmax(fmax(v_a[1], v_b[1]), v_c[1]);
  aabb->max[2] = fmax(fmax(v_a[2], v_b[2]), v_c[2]);
}

void gp_aabb_merge(const gp_aabb* aabb_a, const gp_aabb* aabb_b, gp_aabb* aabb_c)
{
  aabb_c->min[0] = fmin(aabb_a->min[0], aabb_b->min[0]);
  aabb_c->min[1] = fmin(aabb_a->min[1], aabb_b->min[1]);
  aabb_c->min[2] = fmin(aabb_a->min[2], aabb_b->min[2]);
  aabb_c->max[0] = fmax(aabb_a->max[0], aabb_b->max[0]);
  aabb_c->max[1] = fmax(aabb_a->max[1], aabb_b->max[1]);
  aabb_c->max[2] = fmax(aabb_a->max[2], aabb_b->max[2]);
}

void gp_aabb_intersect(
  const gp_aabb* aabb_a,
  const gp_aabb* aabb_b,
  gp_aabb* aabb_c)
{
  aabb_c->min[0] = fmax(aabb_a->min[0], aabb_b->min[0]);
  aabb_c->min[1] = fmax(aabb_a->min[1], aabb_b->min[1]);
  aabb_c->min[2] = fmax(aabb_a->min[2], aabb_b->min[2]);
  aabb_c->max[0] = fmin(aabb_a->max[0], aabb_b->max[0]);
  aabb_c->max[1] = fmin(aabb_a->max[1], aabb_b->max[1]);
  aabb_c->max[2] = fmin(aabb_a->max[2], aabb_b->max[2]);
}

void gp_aabb_size(const gp_aabb* aabb, gp_vec3 size)
{
  size[0] = aabb->max[0] - aabb->min[0];
  size[1] = aabb->max[1] - aabb->min[1];
  size[2] = aabb->max[2] - aabb->min[2];
}

float gp_aabb_half_area(const gp_aabb* aabb)
{
  gp_vec3 size;
  gp_aabb_size(aabb, size);

  return size[0] * size[1] + size[0] * size[2] + size[1] * size[2];
}

float gp_aabb_area(const gp_aabb* aabb)
{
  return 2.0f * gp_aabb_half_area(aabb);
}
