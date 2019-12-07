#include "bvh.h"

#include <math.h>

typedef struct aabb {
  float min_x;
  float min_y;
  float min_z;
  float max_x;
  float max_y;
  float max_z;
} aabb;

typedef struct triangle_ref {
  aabb aabb;
  uint32_t index;
  uint32_t padding;
} triangle_ref;

inline __attribute__((hot, always_inline))
float half_surface_area(const aabb* aabb)
{
  const float d_x = aabb->max_x - aabb->min_x;
  const float d_y = aabb->max_y - aabb->min_y;
  const float d_z = aabb->max_z - aabb->min_z;
  return d_x * d_y + d_x * d_z + d_y * d_z;
}

inline __attribute__((hot, always_inline))
float min_f(float f1, float f2) {
  return f1 < f2 ? f1 : f2;
}

inline __attribute__((hot, always_inline))
float max_f(float f1, float f2) {
  return f1 > f2 ? f1 : f2;
}

inline __attribute__((hot, always_inline))
void gp_bvh_aabb_grow(aabb* aabb1, const aabb* aabb2)
{
  aabb1->min_x = min_f(aabb1->min_x, aabb2->min_x);
  aabb1->min_y = min_f(aabb1->min_y, aabb2->min_y);
  aabb1->min_z = min_f(aabb1->min_z, aabb2->min_z);
  aabb1->max_x = max_f(aabb1->max_x, aabb2->max_x);
  aabb1->max_y = max_f(aabb1->max_y, aabb2->max_y);
  aabb1->max_z = max_f(aabb1->max_z, aabb2->max_z);
}

GpResult gp_bvh_collapse(gp_bvh* bvh, uint32_t k)
{
  // TODO

  return GP_OK;
}

GpResult gp_bvh_build(const gp_bvh_build_input* input, gp_bvh* bvh)
{
  // TODO

  const GpResult result = gp_bvh_collapse(bvh, 3);
  if (result != GP_OK) {
    return result;
  }

  return GP_OK;
}
