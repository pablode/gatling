#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <string_view>

namespace sg
{
  class IShaderCompiler
  {
  public:
    IShaderCompiler(std::string_view shaderPath);

    virtual ~IShaderCompiler();

  public:
    virtual bool init() = 0;

    virtual bool compileHlslToSpv(std::string_view source,
                                  std::string_view filePath,
                                  std::string_view entryPoint,
                                  std::vector<uint8_t>& spv) = 0;

  protected:
    const std::string m_shaderPath;
  };
}
