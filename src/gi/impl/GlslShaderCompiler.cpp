//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "GlslShaderCompiler.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

#include <fstream>
#include <gtl/gb/Log.h>

namespace
{
  using ShaderStage = gtl::GiGlslShaderCompiler::ShaderStage;

  EShLanguage _GetGlslangShaderLanguage(ShaderStage stage)
  {
    switch (stage)
    {
    case ShaderStage::AnyHit:     return EShLangAnyHit;
    case ShaderStage::ClosestHit: return EShLangClosestHit;
    case ShaderStage::Compute:    return EShLangCompute;
    case ShaderStage::Miss:       return EShLangMiss;
    case ShaderStage::RayGen:     return EShLangRayGen;
    default:
      assert(false);
      return EShLangCount;
    }
  }
}

namespace gtl
{
  class _FileIncluder : public glslang::TShader::Includer
  {
  private:
    fs::path m_rootPath;

  public:
    _FileIncluder(const fs::path& rootPath)
      : m_rootPath(rootPath)
    {
    }

    IncludeResult* includeSystem([[maybe_unused]] const char* headerName,
                                 [[maybe_unused]] const char* includerName,
                                 [[maybe_unused]] size_t inclusionDepth) override
    {
      // There's no reason to support this right now.
      return nullptr;
    }

    IncludeResult* includeLocal(const char* headerName,
                                [[maybe_unused]] const char* includerName,
                                [[maybe_unused]] size_t inclusionDepth) override
    {
      fs::path filePath = m_rootPath / headerName;

      std::ifstream fileStream(filePath.c_str(), std::ios_base::binary | std::ios_base::ate);
      if (!fileStream.is_open())
      {
        std::string pathStr = filePath.string();
        GB_ERROR("failed to find shader include '{}'", pathStr);
        return nullptr;
      }

      size_t textLength = fileStream.tellg();
      char* text = new char[textLength];
      fileStream.seekg(0, std::ios::beg);
      fileStream.read(text, textLength);
      return new IncludeResult(headerName, text, textLength, text);
    }

    void releaseInclude(IncludeResult* result) override
    {
      if (!result)
      {
        return;
      }
      delete[] static_cast<char*>(result->userData);
      delete result;
    }
  };

  GiGlslShaderCompiler::GiGlslShaderCompiler(const fs::path& shaderPath)
    : m_fileIncluder(std::make_shared<_FileIncluder>(shaderPath))
  {
    // glslang requires this static initialization, however it internally
    // ref-counts and is thread-safe. The return value seems to be unused.
    [[maybe_unused]] int r = glslang::InitializeProcess();
  }

  GiGlslShaderCompiler::~GiGlslShaderCompiler()
  {
    glslang::FinalizeProcess(); // see above
  }

  bool GiGlslShaderCompiler::compileGlslToSpv(ShaderStage stage,
                                               std::string_view source,
                                               std::vector<uint8_t>& spv)
  {
    EShLanguage language = _GetGlslangShaderLanguage(stage);

    glslang::TShader shader(language);

    const char* sources[] = { source.data() };
    const int sourceLengths[] = { static_cast<int>(source.length()) };
    shader.setStringsWithLengths(sources, sourceLengths, 1);
    shader.setEntryPoint("main");
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EshTargetClientVersion::EShTargetVulkan_1_1);
    shader.setEnvTarget(glslang::EShTargetLanguage::EShTargetSpv, glslang::EShTargetLanguageVersion::EShTargetSpv_1_4);
    shader.setEnvInput(glslang::EShSourceGlsl, language, glslang::EShClient::EShClientVulkan, 450);

    EShMessages messages = static_cast<EShMessages>(EShMsgVulkanRules | EShMsgSpvRules);
#ifndef NDEBUG
    messages = static_cast<EShMessages>(messages | EShMsgDebugInfo);
#endif

    const TBuiltInResource* resourceLimits = GetDefaultResources(); // TODO: do we want to use the actual device limits?
    int defaultVersion = 450; // Will be overriden by #version in source.
    bool forwardCompatible = false;

    bool success = shader.parse(resourceLimits, defaultVersion, forwardCompatible, messages, *m_fileIncluder);
    if (!success)
    {
      const char* msgDesc = "failed to compile shader";
#ifndef NDEBUG
      GB_ERROR("{}: {} ({})", msgDesc, shader.getInfoLog(), shader.getInfoDebugLog());
#else
      GB_ERROR("{}: {}", msgDesc, shader.getInfoLog());
#endif
      return false;
    }

    glslang::TProgram program;
    program.addShader(&shader);

    success = program.link(messages);
    if (!success)
    {
      GB_ERROR("failed to link program");
      return false;
    }

    glslang::SpvOptions spvOptions;
#ifdef NDEBUG
    spvOptions.stripDebugInfo = true;
#else
    spvOptions.generateDebugInfo = true;
    spvOptions.validate = true;
#endif

    glslang::TIntermediate* intermediate = program.getIntermediate(language);
    glslang::GlslangToSpv(*intermediate, *reinterpret_cast<std::vector<unsigned int>*>(&spv), &spvOptions);
    return true;
  }
}
