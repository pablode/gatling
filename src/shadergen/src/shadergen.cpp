#include "shadergen.h"

#include <shaderc/shaderc.h>
#include <shaderc/env.h>
#include <libshaderc_util/file_finder.h>

#include <memory>
#include <string>
#include <vector>
#include <fstream>

std::string s_shadersPath;
shaderc_compiler_t s_compiler;
shaderc_compile_options_t s_compileOptions;
shaderc_util::FileFinder s_fileFinder;

bool _sgReadTextFromFile(const std::string& filePath,
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

namespace sg
{
  struct IncludeStringKeeper
  {
    std::string sourcePath;
    std::string content;
  };
}

shaderc_include_result* _sgResolveInclude(void* userData,
                                          const char* requestedSource,
                                          int inclusionType,
                                          const char* requestingSource,
                                          size_t includeDepth)
{
  std::string sourcePath;
  if (inclusionType == shaderc_include_type_relative)
  {
    sourcePath = s_fileFinder.FindRelativeReadableFilepath(requestingSource, requestedSource);
  }
  else
  {
    sourcePath = s_fileFinder.FindReadableFilepath(requestedSource);
  }

  std::string content;
  if (!_sgReadTextFromFile(sourcePath, content))
  {
    return nullptr;
  }

  auto keeper = new sg::IncludeStringKeeper();
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

void _sgReleaseInclude(void* user_data,
                       shaderc_include_result* includeResult)
{
  auto keeper = reinterpret_cast<sg::IncludeStringKeeper*>(includeResult->user_data);
  delete keeper;
  delete includeResult;
}

bool sgInitialize(const char* resourcePath)
{
  s_shadersPath = std::string(resourcePath) + "/shaders";

  s_compiler = shaderc_compiler_initialize();
  if (!s_compiler)
  {
    return false;
  }

  s_compileOptions = shaderc_compile_options_initialize();
  if (!s_compileOptions)
  {
    shaderc_compiler_release(s_compiler);
    return false;
  }

  shaderc_compile_options_set_source_language(s_compileOptions, shaderc_source_language_hlsl);
  shaderc_compile_options_set_optimization_level(s_compileOptions, shaderc_optimization_level_performance);
  shaderc_compile_options_set_target_env(s_compileOptions, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
  shaderc_compile_options_set_target_spirv(s_compileOptions, shaderc_spirv_version_1_3);

  shaderc_compile_options_set_include_callbacks(
    s_compileOptions,
    _sgResolveInclude,
    _sgReleaseInclude,
    nullptr
  );

  return true;
}

void sgTerminate()
{
  shaderc_compile_options_release(s_compileOptions);
  shaderc_compiler_release(s_compiler);
}

bool _sgCompileHlslToSpv(const std::string& shaderPath,
                         const char* entryPoint,
                         uint32_t* spvSize,
                         uint32_t** spv)
{
  std::string source;
  if (!_sgReadTextFromFile(shaderPath, source))
  {
    return false;
  }

  shaderc_compilation_result_t result = shaderc_compile_into_spv(
    s_compiler,
    source.c_str(),
    source.size(),
    shaderc_compute_shader,
    shaderPath.c_str(),
    entryPoint,
    s_compileOptions
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

bool sgGenerateMainShader(uint32_t* spvSize,
                          uint32_t** spv)
{
  std::string shaderPath = s_shadersPath + "/main.comp.hlsl";
  const char* entryPoint = "CSMain";

  return _sgCompileHlslToSpv(shaderPath, entryPoint, spvSize, spv);
}
