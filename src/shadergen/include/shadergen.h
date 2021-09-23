#ifndef SHADERGEN_H
#define SHADERGEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool sgInitialize(const char* resourcePath);

void sgTerminate();

struct SgMainShaderParams
{
  uint32_t num_threads_x;
  uint32_t num_threads_y;
  uint32_t max_stack_size;
  uint32_t image_width;
  uint32_t image_height;
  uint32_t spp;
  uint32_t max_bounces;
  float camera_position_x;
  float camera_position_y;
  float camera_position_z;
  float camera_forward_x;
  float camera_forward_y;
  float camera_forward_z;
  float camera_up_x;
  float camera_up_y;
  float camera_up_z;
  float camera_vfov;
  uint32_t rr_bounce_offset;
  float rr_inv_min_term_prob;
};

bool sgGenerateMainShader(const struct SgMainShaderParams* params,
                          uint32_t* spvSize,
                          uint32_t** spv,
                          const char** entryPoint);

#ifdef __cplusplus
}
#endif

#endif
