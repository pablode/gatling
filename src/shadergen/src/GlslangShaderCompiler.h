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

    bool compileHlslToSpv(std::string_view source,
                          std::string_view filePath,
                          std::string_view entryPoint,
                          std::vector<uint8_t>& spv) override;

  private:
    shaderc_compiler_t m_compiler = nullptr;
    shaderc_compile_options_t m_compileOptions = nullptr;
  };
}
