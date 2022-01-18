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
    DxcShaderCompiler(std::string_view shaderPath);

  public:
    bool init() override;

    bool compileHlslToSpv(std::string_view source,
                          std::string_view filePath,
                          std::string_view entryPoint,
                          std::vector<uint8_t>& spv) override;

  private:
    CComPtr<IDxcCompiler3> m_compiler;
    CComPtr<IDxcIncludeHandler> m_includeHandler;
    CComPtr<IDxcUtils> m_utils;
  };
}
