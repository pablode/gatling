#include "aabb.h"

#include <math.h>
#include <assert.h>

void gp_vec3_assign(const gp_vec3 a, gp_vec3 b)
{
  b[0] = a[0];
  b[1] = a[1];
  b[2] = a[2];
}

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

void gp_vec3_div(const gp_vec3 a, const gp_vec3 b, gp_vec3 c)
{
  c[0] = a[0] / b[0];
  c[1] = a[1] / b[1];
  c[2] = a[2] / b[2];
}

void gp_vec3_divs(const gp_vec3 a, float s, gp_vec3 b)
{
  assert(s != 0.0f);
  b[0] = a[0] / s;
  b[1] = a[1] / s;
  b[2] = a[2] / s;
}

void gp_vec3_sdiv(float s, const gp_vec3 a, gp_vec3 b)
{
  assert(a[0] != 0.0f);
  assert(a[1] != 0.0f);
  assert(a[2] != 0.0f);
  b[0] = s / a[0];
  b[1] = s / a[1];
  b[2] = s / a[2];
}

void gp_vec3_mul(const gp_vec3 a, const gp_vec3 b, gp_vec3 c)
{
  c[0] = a[0] * b[0];
  c[1] = a[1] * b[1];
  c[2] = a[2] * b[2];
}

void gp_vec3_muls(const gp_vec3 a, float s, gp_vec3 b)
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
  v[0] = (1.0f - t) * a[0] + t * b[0];
  v[1] = (1.0f - t) * a[1] + t * b[1];
  v[2] = (1.0f - t) * a[2] + t * b[2];
}

float gp_vec3_length(const gp_vec3 v)
{
  return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void gp_vec3_normalize(const gp_vec3 a, gp_vec3 b)
{
  float inv_length = 1.0f / gp_vec3_length(a);
  b[0] = a[0] * inv_length;
  b[1] = a[1] * inv_length;
  b[2] = a[2] * inv_length;
}

void gp_vec3_max(const gp_vec3 a, const gp_vec3 b, gp_vec3 c)
{
  c[0] = fmaxf(a[0], b[0]);
  c[1] = fmaxf(a[1], b[1]);
  c[2] = fmaxf(a[2], b[2]);
}

void gp_vec3_min(const gp_vec3 a, const gp_vec3 b, gp_vec3 c)
{
  c[0] = fminf(a[0], b[0]);
  c[1] = fminf(a[1], b[1]);
  c[2] = fminf(a[2], b[2]);
}

float gp_vec3_comp_min(const gp_vec3 v)
{
  return fminf(fminf(v[0], v[1]), v[2]);
}

float gp_vec3_comp_max(const gp_vec3 v)
{
  return fmaxf(fmaxf(v[0], v[1]), v[2]);
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

void gp_aabb_include(const gp_aabb* aabb_a, const gp_vec3 v, gp_aabb* aabb_b)
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

void gp_aabb_size(const gp_aabb* aabb, gp_vec3 size)
{
  size[0] = fmaxf(0.0f, aabb->max[0] - aabb->min[0]);
  size[1] = fmaxf(0.0f, aabb->max[1] - aabb->min[1]);
  size[2] = fmaxf(0.0f, aabb->max[2] - aabb->min[2]);
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
