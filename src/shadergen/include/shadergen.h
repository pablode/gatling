#ifndef SHADERGEN_H
#define SHADERGEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SgMaterial;

bool sgInitialize(const char* resourcePath,
                  const char* shaderPath,
                  const char* mtlxlibPath,
                  const char* mtlxmdlPath);

void sgTerminate();

struct SgMaterial* sgCreateMaterialFromMtlx(const char* docStr);

void sgDestroyMaterial(struct SgMaterial* mat);

struct SgMainShaderParams
{
  uint32_t num_threads_x;
  uint32_t num_threads_y;
  uint32_t max_stack_size;
  uint32_t spp;
  uint32_t max_bounces;
  uint32_t rr_bounce_offset;
  float rr_inv_min_term_prob;
  float max_sample_value;
  float bg_color[4];
  uint32_t material_count;
  const struct SgMaterial** materials;
};

bool sgGenerateMainShader(const struct SgMainShaderParams* params,
                          uint32_t* spvSize,
                          uint32_t** spv,
                          const char** entryPoint);

#ifdef __cplusplus
}
#endif

#endif
