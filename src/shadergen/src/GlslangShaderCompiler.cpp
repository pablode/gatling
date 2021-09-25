#include "GlslangShaderCompiler.h"

#include <shaderc/env.h>
#include <libshaderc_util/file_finder.h>

#include <fstream>

namespace detail
{
  bool readTextFromFile(const std::string& filePath,
                        std::string& text)
  {
    std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
    if (!file.is_open())
    {
      return false;
    }
    file.seekg(0, std::ios_base::end);
    text.resize(file.tellg(), ' ');
    file.seekg(0, std::ios_base::beg);
    file.read(&text[0], text.size());
    return file.good();
  }

  struct IncludeStringKeeper
  {
    std::string sourcePath;
    std::string content;
  };

  shaderc_include_result* resolveInclude(void* userData,
                                         const char* requestedSource,
                                         int inclusionType,
                                         const char* requestingSource,
                                         size_t includeDepth)
  {
    std::string sourcePath;
    shaderc_util::FileFinder fileFinder;
    if (inclusionType == shaderc_include_type_relative)
    {
      sourcePath = fileFinder.FindRelativeReadableFilepath(requestingSource, requestedSource);
    }
    else
    {
      sourcePath = fileFinder.FindReadableFilepath(requestedSource);
    }

    std::string content;
    if (!readTextFromFile(sourcePath, content))
    {
      return nullptr;
    }

    auto keeper = new IncludeStringKeeper();
    keeper->sourcePath = sourcePath;
    keeper->content = content;

    auto result = new shaderc_include_result();
    result->user_data = keeper;
    result->source_name = keeper->sourcePath.c_str();
    result->source_name_length = keeper->sourcePath.size();
    result->content = keeper->content.c_str();
    result->content_length = keeper->content.size();
    return result;
  }

  void releaseInclude(void* user_data,
                      shaderc_include_result* includeResult)
  {
    auto keeper = reinterpret_cast<IncludeStringKeeper*>(includeResult->user_data);
    delete keeper;
    delete includeResult;
  }
}

namespace sg
{
  GlslangShaderCompiler::GlslangShaderCompiler(const std::string& shaderPath)
    : IShaderCompiler(shaderPath)
  {
  }

  bool GlslangShaderCompiler::init()
  {
    m_compiler = shaderc_compiler_initialize();
    if (!m_compiler)
    {
      return false;
    }

    m_compileOptions = shaderc_compile_options_initialize();
    if (!m_compileOptions)
    {
      shaderc_compiler_release(m_compiler);
      return false;
    }

    shaderc_compile_options_set_source_language(m_compileOptions, shaderc_source_language_hlsl);
    shaderc_compile_options_set_optimization_level(m_compileOptions, shaderc_optimization_level_performance);
    shaderc_compile_options_set_target_env(m_compileOptions, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    shaderc_compile_options_set_target_spirv(m_compileOptions, shaderc_spirv_version_1_3);

    shaderc_compile_options_set_include_callbacks(
      m_compileOptions,
      detail::resolveInclude,
      detail::releaseInclude,
      nullptr
    );

    return true;
  }

  GlslangShaderCompiler::~GlslangShaderCompiler()
  {
    if (m_compileOptions)
    {
      shaderc_compile_options_release(m_compileOptions);
    }
    if (m_compiler)
    {
      shaderc_compiler_release(m_compiler);
    }
  }

  bool GlslangShaderCompiler::compileHlslToSpv(const std::string& source,
                                               const std::string& filePath,
                                               const char* entryPoint,
                                               uint32_t* spvSize,
                                               uint32_t** spv)
  {
    shaderc_compilation_result_t result = shaderc_compile_into_spv(
      m_compiler,
      source.c_str(),
      source.size(),
      shaderc_compute_shader,
      filePath.c_str(),
      entryPoint,
      m_compileOptions
    );

    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success)
    {
      shaderc_result_release(result);
      fprintf(stderr, "Failed to compile shader: %s\n", shaderc_result_get_error_message(result));
      return false;
    }

    *spvSize = shaderc_result_get_length(result);

    const char* data = shaderc_result_get_bytes(result);
    *spv = (uint32_t*) malloc(*spvSize);
    memcpy(*spv, data, *spvSize);

    shaderc_result_release(result);

    return true;
  }
}
