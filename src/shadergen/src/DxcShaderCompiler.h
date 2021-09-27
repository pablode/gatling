#pragma once

#include "IShaderCompiler.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <atlbase.h>
#include <dxcapi.h>

namespace sg
{
  class DxcShaderCompiler : public IShaderCompiler
  {
  public:
    DxcShaderCompiler(const std::string& shaderPath);

  public:
    bool init() override;

    bool compileHlslToSpv(const std::string& source,
                          const std::string& filePath,
                          const char* entryPoint,
                          uint32_t* spvSize,
                          uint32_t** spv) override;

  private:
    CComPtr<IDxcCompiler3> m_compiler;
    CComPtr<IDxcIncludeHandler> m_includeHandler;
    CComPtr<IDxcUtils> m_utils;
  };
}
