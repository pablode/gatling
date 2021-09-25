#pragma once

#include "IShaderCompiler.h"

#include <shaderc/shaderc.h>

namespace sg
{
  class GlslangShaderCompiler : public IShaderCompiler
  {
  public:
    GlslangShaderCompiler(const std::string& shaderPath);

    ~GlslangShaderCompiler();

  public:
    bool init() override;

    bool compileHlslToSpv(const std::string& source,
                          const std::string& filePath,
                          const char* entryPoint,
                          uint32_t* spvSize,
                          uint32_t** spv) override;

  private:
    shaderc_compiler_t m_compiler = nullptr;
    shaderc_compile_options_t m_compileOptions = nullptr;
  };
}
