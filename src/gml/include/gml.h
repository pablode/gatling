#ifndef GML_H
#define GML_H

#include <stdint.h>

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
void gml_vec3_sdiv(float s, const gml_vec3 a, gml_vec3 b);
void gml_vec3_mul(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_muls(const gml_vec3 a, float s, gml_vec3 b);
float gml_vec3_dot(const gml_vec3 a, const gml_vec3 b);
void gml_vec3_cross(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_lerp(const gml_vec3 a, const gml_vec3 b, float t, gml_vec3 v);
float gml_vec3_length(const gml_vec3 v);
void gml_vec3_normalize(const gml_vec3 a, gml_vec3 b);
void gml_vec3_max(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
void gml_vec3_min(const gml_vec3 a, const gml_vec3 b, gml_vec3 c);
float gml_vec3_comp_min(const gml_vec3 v);
float gml_vec3_comp_max(const gml_vec3 v);

#endif
