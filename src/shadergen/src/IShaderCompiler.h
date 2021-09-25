#pragma once

#include <stdint.h>
#include <string>

namespace sg
{
  class IShaderCompiler
  {
  public:
    IShaderCompiler(const std::string& shaderPath);

    virtual ~IShaderCompiler();

  public:
    virtual bool init() = 0;

    virtual bool compileHlslToSpv(const std::string& source,
                                  const std::string& filePath,
                                  const char* entryPoint,
                                  uint32_t* spvSize,
                                  uint32_t** spv) = 0;

  protected:
    const std::string m_shaderPath;
  };
}
