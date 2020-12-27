#include "gml.h"

#include <math.h>
#include <assert.h>

#define GML_MIN_FUNC(PREFIX, TYPE)   \
  TYPE PREFIX##min(TYPE a, TYPE b) { \
    return (a < b) ? a : b;          \
  }
#define GML_MAX_FUNC(PREFIX, TYPE)   \
  TYPE PREFIX##max(TYPE a, TYPE b) { \
    return (a > b) ? a : b;          \
  }

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

void gml_vec3_assign(const gml_vec3 a, gml_vec3 b)
{
  b[0] = a[0];
  b[1] = a[1];
  b[2] = a[2];
}

void gml_vec3_add(const gml_vec3 a, const gml_vec3 b, gml_vec3 c)
{
  c[0] = a[0] + b[0];
  c[1] = a[1] + b[1];
  c[2] = a[2] + b[2];
}

void gml_vec3_sub(const gml_vec3 a, const gml_vec3 b, gml_vec3 c)
{
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] - b[2];
}

void gml_vec3_div(const gml_vec3 a, const gml_vec3 b, gml_vec3 c)
{
  c[0] = a[0] / b[0];
  c[1] = a[1] / b[1];
  c[2] = a[2] / b[2];
}

void gml_vec3_divs(const gml_vec3 a, float s, gml_vec3 b)
{
  assert(s != 0.0f);
  b[0] = a[0] / s;
  b[1] = a[1] / s;
  b[2] = a[2] / s;
}

void gml_vec3_mul(const gml_vec3 a, const gml_vec3 b, gml_vec3 c)
{
  c[0] = a[0] * b[0];
  c[1] = a[1] * b[1];
  c[2] = a[2] * b[2];
}

void gml_vec3_muls(const gml_vec3 a, float s, gml_vec3 b)
{
  b[0] = a[0] * s;
  b[1] = a[1] * s;
  b[2] = a[2] * s;
}

float gml_vec3_dot(const gml_vec3 a, const gml_vec3 b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void gml_vec3_cross(const gml_vec3 a, const gml_vec3 b, gml_vec3 c)
{
  c[0] = a[1] * b[2] - b[1] * a[2];
  c[1] = a[2] * b[0] - b[2] * a[0];
  c[2] = a[0] * b[1] - b[0] * a[1];
}

void gml_vec3_lerp(const gml_vec3 a, const gml_vec3 b, float t, gml_vec3 v)
{
  v[0] = (1.0f - t) * a[0] + t * b[0];
  v[1] = (1.0f - t) * a[1] + t * b[1];
  v[2] = (1.0f - t) * a[2] + t * b[2];
}

float gml_vec3_length(const gml_vec3 v)
{
  return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void gml_vec3_normalize(const gml_vec3 a, gml_vec3 b)
{
  float inv_length = 1.0f / gml_vec3_length(a);
  b[0] = a[0] * inv_length;
  b[1] = a[1] * inv_length;
  b[2] = a[2] * inv_length;
}

void gml_vec3_max(const gml_vec3 a, const gml_vec3 b, gml_vec3 c)
{
  c[0] = fmaxf(a[0], b[0]);
  c[1] = fmaxf(a[1], b[1]);
  c[2] = fmaxf(a[2], b[2]);
}

void gml_vec3_min(const gml_vec3 a, const gml_vec3 b, gml_vec3 c)
{
  c[0] = fminf(a[0], b[0]);
  c[1] = fminf(a[1], b[1]);
  c[2] = fminf(a[2], b[2]);
}

float gml_vec3_mincomp(const gml_vec3 v)
{
  return fminf(fminf(v[0], v[1]), v[2]);
}

float gml_vec3_maxcomp(const gml_vec3 v)
{
  return fmaxf(fmaxf(v[0], v[1]), v[2]);
}

void gml_vec4_assign(const gml_vec4 a, gml_vec4 b)
{
  b[0] = a[0];
  b[1] = a[1];
  b[2] = a[2];
  b[3] = a[3];
}

void gml_mat4_assign(const gml_mat4 a, gml_mat4 b)
{
  b[0][0] = a[0][0];
  b[0][1] = a[0][1];
  b[0][2] = a[0][2];
  b[0][3] = a[0][3];
  b[1][0] = a[1][0];
  b[1][1] = a[1][1];
  b[1][2] = a[1][2];
  b[1][3] = a[1][3];
  b[2][0] = a[2][0];
  b[2][1] = a[2][1];
  b[2][2] = a[2][2];
  b[2][3] = a[2][3];
  b[3][0] = a[3][0];
  b[3][1] = a[3][1];
  b[3][2] = a[3][2];
  b[3][3] = a[3][3];
}

void gml_mat4_mul(const gml_mat4 a, const gml_mat4 b, gml_mat4 c)
{
  gml_mat4 tmp;
  tmp[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0];
  tmp[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1];
  tmp[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2];
  tmp[0][3] = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3];
  tmp[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0];
  tmp[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1];
  tmp[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2];
  tmp[1][3] = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3];
  tmp[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0];
  tmp[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1];
  tmp[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2];
  tmp[2][3] = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3];
  tmp[3][0] = a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] + a[3][3] * b[3][0];
  tmp[3][1] = a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] + a[3][3] * b[3][1];
  tmp[3][2] = a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] + a[3][3] * b[3][2];
  tmp[3][3] = a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] + a[3][3] * b[3][3];
  gml_mat4_assign(tmp, c);
}

void gml_mat4_mul_vec4(const gml_mat4 a, const gml_vec4 b, gml_vec4 c)
{
  gml_vec4 tmp;
  tmp[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2] + a[0][3] * b[3];
  tmp[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2] + a[1][3] * b[3];
  tmp[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2] + a[2][3] * b[3];
  tmp[3] = a[3][0] * b[0] + a[3][1] * b[1] + a[3][2] * b[2] + a[3][3] * b[3];
  gml_vec4_assign(tmp, c);
}

void gml_mat4_identity(gml_mat4 a)
{
  a[0][0] = 1.0f;
  a[0][1] = 0.0f;
  a[0][2] = 0.0f;
  a[0][3] = 0.0f;
  a[1][0] = 0.0f;
  a[1][1] = 1.0f;
  a[1][2] = 0.0f;
  a[1][3] = 0.0f;
  a[2][0] = 0.0f;
  a[2][1] = 0.0f;
  a[2][2] = 1.0f;
  a[2][3] = 0.0f;
  a[3][0] = 0.0f;
  a[3][1] = 0.0f;
  a[3][2] = 0.0f;
  a[3][3] = 1.0f;
}

void gml_mat3_assign(const gml_mat3 a, gml_mat3 b)
{
  b[0][0] = a[0][0];
  b[0][1] = a[0][1];
  b[0][2] = a[0][2];
  b[1][0] = a[1][0];
  b[1][1] = a[1][1];
  b[1][2] = a[1][2];
  b[2][0] = a[2][0];
  b[2][1] = a[2][1];
  b[2][2] = a[2][2];
}

void gml_mat3_from_mat4(const gml_mat4 a, gml_mat3 b)
{
  b[0][0] = a[0][0];
  b[0][1] = a[0][1];
  b[0][2] = a[0][2];
  b[1][0] = a[1][0];
  b[1][1] = a[1][1];
  b[1][2] = a[1][2];
  b[2][0] = a[2][0];
  b[2][1] = a[2][1];
  b[2][2] = a[2][2];
}

bool gml_mat3_invert(const gml_mat3 a, gml_mat3 b)
{
  float det = + a[0][0] * a[1][1] * a[2][2]
              - a[0][0] * a[1][2] * a[2][1]
              + a[0][1] * a[1][2] * a[2][0]
              - a[0][1] * a[1][0] * a[2][2]
              + a[0][2] * a[1][0] * a[2][1]
              - a[0][2] * a[1][1] * a[2][0];

  if (det == 0.0f)
  {
    /* Matrix is not invertible. */
    return false;
  }

  float idet = 1.0f / det;

  gml_mat3 tmp;
  tmp[0][0] = +idet * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
  tmp[0][1] = -idet * (a[0][1] * a[2][2] - a[0][2] * a[2][1]);
  tmp[0][2] = +idet * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);
  tmp[1][0] = -idet * (a[1][0] * a[2][2] - a[1][2] * a[2][0]);
  tmp[1][1] = +idet * (a[0][0] * a[2][2] - a[0][2] * a[2][0]);
  tmp[1][2] = -idet * (a[0][0] * a[1][2] - a[0][2] * a[1][0]);
  tmp[2][0] = +idet * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  tmp[2][1] = -idet * (a[0][0] * a[2][1] - a[0][1] * a[2][0]);
  tmp[2][2] = +idet * (a[0][0] * a[1][1] - a[0][1] * a[1][0]);

  gml_mat3_assign(tmp, b);

  return true;
}

void gml_mat3_transpose(const gml_mat3 a, gml_mat3 b)
{
  gml_mat3 tmp;
  tmp[0][0] = a[0][0];
  tmp[0][1] = a[1][0];
  tmp[0][2] = a[2][0];
  tmp[1][0] = a[0][1];
  tmp[1][1] = a[1][1];
  tmp[1][2] = a[2][1];
  tmp[2][0] = a[0][2];
  tmp[2][1] = a[1][2];
  tmp[2][2] = a[2][2];
  gml_mat3_assign(tmp, b);
}

void gml_mat3_mul_vec3(const gml_mat3 a, const gml_vec3 b, gml_vec3 c)
{
  gml_vec3 tmp;
  tmp[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2];
  tmp[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2];
  tmp[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2];
  gml_vec3_assign(tmp, c);
}
