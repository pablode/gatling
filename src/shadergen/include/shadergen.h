#ifndef SHADERGEN_H
#define SHADERGEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool sgInitialize(const char* resourcePath);

void sgTerminate();

bool sgGenerateMainShader(uint32_t* spvSize,
                          uint32_t** spv);

#ifdef __cplusplus
}
#endif

#endif
