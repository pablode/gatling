#ifndef GML_H
#define GML_H

#include <stdint.h>

typedef float gp_vec3[3];

void gp_vec3_assign(const gp_vec3 a, gp_vec3 b);
void gp_vec3_add(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
void gp_vec3_sub(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
void gp_vec3_div(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
void gp_vec3_divs(const gp_vec3 a, float s, gp_vec3 b);
void gp_vec3_sdiv(float s, const gp_vec3 a, gp_vec3 b);
void gp_vec3_mul(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
void gp_vec3_muls(const gp_vec3 a, float s, gp_vec3 b);
float gp_vec3_dot(const gp_vec3 a, const gp_vec3 b);
void gp_vec3_cross(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
void gp_vec3_lerp(const gp_vec3 a, const gp_vec3 b, float t, gp_vec3 v);
float gp_vec3_length(const gp_vec3 v);
void gp_vec3_normalize(const gp_vec3 a, gp_vec3 b);
void gp_vec3_max(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
void gp_vec3_min(const gp_vec3 a, const gp_vec3 b, gp_vec3 c);
float gp_vec3_comp_min(const gp_vec3 v);
float gp_vec3_comp_max(const gp_vec3 v);

#endif
