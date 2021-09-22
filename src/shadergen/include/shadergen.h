#ifndef SHADERGEN_H
#define SHADERGEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool sgGenerateMainShader(uint32_t* spv_size,
                          uint32_t** spv);

#ifdef __cplusplus
}
#endif

#endif
