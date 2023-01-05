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

#include "GlslangShaderCompiler.h"

#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>

#include <fstream>

namespace detail
{
  // https://github.com/KhronosGroup/glslang/blob/master/StandAlone/ResourceLimits.cpp
  // FIXME: use actual device limits?
  const static TBuiltInResource s_defaultResourceLimits = {
    /* .MaxLights = */ 32,
    /* .MaxClipPlanes = */ 6,
    /* .MaxTextureUnits = */ 32,
    /* .MaxTextureCoords = */ 32,
    /* .MaxVertexAttribs = */ 64,
    /* .MaxVertexUniformComponents = */ 4096,
    /* .MaxVaryingFloats = */ 64,
    /* .MaxVertexTextureImageUnits = */ 32,
    /* .MaxCombinedTextureImageUnits = */ 80,
    /* .MaxTextureImageUnits = */ 32,
    /* .MaxFragmentUniformComponents = */ 4096,
    /* .MaxDrawBuffers = */ 32,
    /* .MaxVertexUniformVectors = */ 128,
    /* .MaxVaryingVectors = */ 8,
    /* .MaxFragmentUniformVectors = */ 16,
    /* .MaxVertexOutputVectors = */ 16,
    /* .MaxFragmentInputVectors = */ 15,
    /* .MinProgramTexelOffset = */ -8,
    /* .MaxProgramTexelOffset = */ 7,
    /* .MaxClipDistances = */ 8,
    /* .MaxComputeWorkGroupCountX = */ 65535,
    /* .MaxComputeWorkGroupCountY = */ 65535,
    /* .MaxComputeWorkGroupCountZ = */ 65535,
    /* .MaxComputeWorkGroupSizeX = */ 1024,
    /* .MaxComputeWorkGroupSizeY = */ 1024,
    /* .MaxComputeWorkGroupSizeZ = */ 64,
    /* .MaxComputeUniformComponents = */ 1024,
    /* .MaxComputeTextureImageUnits = */ 16,
    /* .MaxComputeImageUniforms = */ 8,
    /* .MaxComputeAtomicCounters = */ 8,
    /* .MaxComputeAtomicCounterBuffers = */ 1,
    /* .MaxVaryingComponents = */ 60,
    /* .MaxVertexOutputComponents = */ 64,
    /* .MaxGeometryInputComponents = */ 64,
    /* .MaxGeometryOutputComponents = */ 128,
    /* .MaxFragmentInputComponents = */ 128,
    /* .MaxImageUnits = */ 8,
    /* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
    /* .MaxCombinedShaderOutputResources = */ 8,
    /* .MaxImageSamples = */ 0,
    /* .MaxVertexImageUniforms = */ 0,
    /* .MaxTessControlImageUniforms = */ 0,
    /* .MaxTessEvaluationImageUniforms = */ 0,
    /* .MaxGeometryImageUniforms = */ 0,
    /* .MaxFragmentImageUniforms = */ 8,
    /* .MaxCombinedImageUniforms = */ 8,
    /* .MaxGeometryTextureImageUnits = */ 16,
    /* .MaxGeometryOutputVertices = */ 256,
    /* .MaxGeometryTotalOutputComponents = */ 1024,
    /* .MaxGeometryUniformComponents = */ 1024,
    /* .MaxGeometryVaryingComponents = */ 64,
    /* .MaxTessControlInputComponents = */ 128,
    /* .MaxTessControlOutputComponents = */ 128,
    /* .MaxTessControlTextureImageUnits = */ 16,
    /* .MaxTessControlUniformComponents = */ 1024,
    /* .MaxTessControlTotalOutputComponents = */ 4096,
    /* .MaxTessEvaluationInputComponents = */ 128,
    /* .MaxTessEvaluationOutputComponents = */ 128,
    /* .MaxTessEvaluationTextureImageUnits = */ 16,
    /* .MaxTessEvaluationUniformComponents = */ 1024,
    /* .MaxTessPatchComponents = */ 120,
    /* .MaxPatchVertices = */ 32,
    /* .MaxTessGenLevel = */ 64,
    /* .MaxViewports = */ 16,
    /* .MaxVertexAtomicCounters = */ 0,
    /* .MaxTessControlAtomicCounters = */ 0,
    /* .MaxTessEvaluationAtomicCounters = */ 0,
    /* .MaxGeometryAtomicCounters = */ 0,
    /* .MaxFragmentAtomicCounters = */ 8,
    /* .MaxCombinedAtomicCounters = */ 8,
    /* .MaxAtomicCounterBindings = */ 1,
    /* .MaxVertexAtomicCounterBuffers = */ 0,
    /* .MaxTessControlAtomicCounterBuffers = */ 0,
    /* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
    /* .MaxGeometryAtomicCounterBuffers = */ 0,
    /* .MaxFragmentAtomicCounterBuffers = */ 1,
    /* .MaxCombinedAtomicCounterBuffers = */ 1,
    /* .MaxAtomicCounterBufferSize = */ 16384,
    /* .MaxTransformFeedbackBuffers = */ 4,
    /* .MaxTransformFeedbackInterleavedComponents = */ 64,
    /* .MaxCullDistances = */ 8,
    /* .MaxCombinedClipAndCullDistances = */ 8,
    /* .MaxSamples = */ 4,
    /* .maxMeshOutputVerticesNV = */ 256,
    /* .maxMeshOutputPrimitivesNV = */ 512,
    /* .maxMeshWorkGroupSizeX_NV = */ 32,
    /* .maxMeshWorkGroupSizeY_NV = */ 1,
    /* .maxMeshWorkGroupSizeZ_NV = */ 1,
    /* .maxTaskWorkGroupSizeX_NV = */ 32,
    /* .maxTaskWorkGroupSizeY_NV = */ 1,
    /* .maxTaskWorkGroupSizeZ_NV = */ 1,
    /* .maxMeshViewCountNV = */ 4,
    /* .maxDualSourceDrawBuffersEXT = */ 1,
    /* .limits = */ {
        /* .nonInductiveForLoops = */ 1,
        /* .whileLoops = */ 1,
        /* .doWhileLoops = */ 1,
        /* .generalUniformIndexing = */ 1,
        /* .generalAttributeMatrixVectorIndexing = */ 1,
        /* .generalVaryingIndexing = */ 1,
        /* .generalSamplerIndexing = */ 1,
        /* .generalVariableIndexing = */ 1,
        /* .generalConstantMatrixVectorIndexing = */ 1,
    }};

  class FileIncluder : public glslang::TShader::Includer
  {
  private:
    std::string m_rootPath;

  public:
    FileIncluder(std::string_view rootPath)
      : m_rootPath(rootPath)
    {
    }

    IncludeResult* includeSystem(const char* headerName,
                                 const char* includerName,
                                 size_t inclusionDepth) override
    {
      // There's no reason to support this right now.
      return nullptr;
    }

    IncludeResult* includeLocal(const char* headerName,
                                const char* includerName,
                                size_t inclusionDepth) override
    {
      std::string fileName = m_rootPath + "/" + headerName;

      std::ifstream fileStream(fileName.c_str(), std::ios_base::binary | std::ios_base::ate);
      if (!fileStream.is_open())
      {
        fprintf(stderr, "Failed to find shader include '%s'\n", fileName.c_str());
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

  using ShaderStage = gi::sg::GlslangShaderCompiler::ShaderStage;

  EShLanguage getGlslangShaderLanguage(ShaderStage stage)
  {
    switch (stage)
    {
    case ShaderStage::Compute:
      return EShLangCompute;
    case ShaderStage::RayGen:
      return EShLangRayGen;
    default:
      assert(false);
      return EShLangCount;
    }
  }
}

namespace gi::sg
{
  static bool s_glslangInitialized = false;

  bool GlslangShaderCompiler::init()
  {
    if (s_glslangInitialized)
    {
      return true;
    }
    return glslang::InitializeProcess();
  }

  void GlslangShaderCompiler::deinit()
  {
    if (!s_glslangInitialized)
    {
      return;
    }
    glslang::FinalizeProcess();
  }

  GlslangShaderCompiler::GlslangShaderCompiler(const std::string& shaderPath)
    : m_fileIncluder(new detail::FileIncluder(shaderPath))
  {
  }

  GlslangShaderCompiler::~GlslangShaderCompiler()
  {
    delete (detail::FileIncluder*) m_fileIncluder;
  }

  bool GlslangShaderCompiler::compileGlslToSpv(ShaderStage stage,
                                               std::string_view source,
                                               std::string_view filePath,
                                               std::vector<uint8_t>& spv)
  {
    EShLanguage language = detail::getGlslangShaderLanguage(stage);

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

    auto printErrorMessage = [&shader](const char* baseFmtString) {
      fprintf(stderr, baseFmtString, shader.getInfoLog());
#ifndef NDEBUG
      const char* debugInfoLog = shader.getInfoDebugLog();
      if (strlen(debugInfoLog) > 0)
      {
        fprintf(stderr, " (%s)", debugInfoLog);
      }
#endif
      fprintf(stderr, "\n");
    };

    int defaultVersion = 450; // Will be overriden by #version in source.
    bool forwardCompatible = false;
    auto fileIncluder = *reinterpret_cast<detail::FileIncluder*>(m_fileIncluder);
    bool success = shader.parse(&detail::s_defaultResourceLimits, defaultVersion, forwardCompatible, messages, fileIncluder);
    if (!success)
    {
      printErrorMessage("Failed to compile shader: %s");
      return false;
    }

    glslang::TProgram program;
    program.addShader(&shader);

    success = program.link(messages);
    if (!success)
    {
      printErrorMessage("Failed to link program: %s");
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
