#include "DxcShaderCompiler.h"

#include <cassert>

namespace sg
{
  DxcShaderCompiler::DxcShaderCompiler(std::string_view shaderPath)
    : IShaderCompiler(shaderPath)
  {
  }

  bool DxcShaderCompiler::init()
  {
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_utils))))
    {
      return false;
    }

    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_compiler))))
    {
      return false;
    }

    if (FAILED(m_utils->CreateDefaultIncludeHandler(&m_includeHandler)))
    {
      return false;
    }

    return true;
  }

  std::wstring _convertCStrToWString(const char* cStr)
  {
    int wstrLen = MultiByteToWideChar(CP_UTF8, 0, cStr, -1, nullptr, 0);
    std::wstring wstr;
    wstr.resize(wstrLen, L' ');
    MultiByteToWideChar(CP_UTF8, 0, cStr, -1, wstr.data(), wstr.size());
    return wstr;
  }

  bool DxcShaderCompiler::compileHlslToSpv(std::string_view source,
                                           std::string_view filePath,
                                           std::string_view entryPoint,
                                           std::vector<uint8_t>& spv)
  {
    std::wstring wFilePath = _convertCStrToWString(filePath.data());
    std::wstring wEntryPoint = _convertCStrToWString(entryPoint.data());

    CComPtr<IDxcBlobEncoding> sourceBufEnc;
    if (FAILED(m_utils->CreateBlob(source.data(), source.size(), DXC_CP_UTF8, &sourceBufEnc)))
    {
      return false;
    }

    DxcBuffer sourceBuf;
    sourceBuf.Ptr = sourceBufEnc->GetBufferPointer();
    sourceBuf.Size = sourceBufEnc->GetBufferSize();
    sourceBuf.Encoding = DXC_CP_UTF8;

    LPCWSTR args[] =
    {
      wFilePath.c_str(),
      L"-E", wEntryPoint.c_str(),
      L"-T", L"cs_6_6",  // Compute shader with Shading Model 6.6
#ifndef NDEBUG
      L"-WX",            // Treat warnings as errors
      L"-Zi",            // Enable debug information.
      L"-Ges",           // Enable strict mode
      L"-Gis",           // Force IEEE strictness
#endif
      // Target SPIR-V instead of DXIL
      L"-spirv",
      L"-fspv-target-env=vulkan1.1"
    };

    CComPtr<IDxcResult> result;
    if (FAILED(m_compiler->Compile(&sourceBuf, args, _countof(args), m_includeHandler, IID_PPV_ARGS(&result))))
    {
      fprintf(stderr, "Internal compilation error\n");
      assert(false);
      return false;
    }

    CComPtr<IDxcBlobUtf8> errorMsgBlob;
    CComPtr<IDxcBlobUtf16> outputName;
    if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errorMsgBlob), &outputName)))
    {
      if (errorMsgBlob && errorMsgBlob->GetStringLength() > 0)
      {
        fprintf(stderr, "Compilation error: %s\n", errorMsgBlob->GetStringPointer());
      }
    }

    HRESULT compStatus;
    if (FAILED(result->GetStatus(&compStatus)) || FAILED(compStatus))
    {
      fprintf(stderr, "Compilation failed\n");
      return false;
    }

    CComPtr<IDxcBlob> spirvBlob;
    if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&spirvBlob), &outputName)))
    {
      return false;
    }

    size_t spvSize = spirvBlob->GetBufferSize();
    spv.resize(spvSize);
    memcpy(spv.data(), spirvBlob->GetBufferPointer(), spvSize);

    return true;
  }
}
