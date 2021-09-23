#include "shadergen.h"

#include <shaderc/shaderc.h>
#include <shaderc/env.h>
#include <libshaderc_util/file_finder.h>

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <limits>

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

bool _sgCompileHlslToSpv(const std::string& source,
                         const std::string& filePath,
                         const char* entryPoint,
                         uint32_t* spvSize,
                         uint32_t** spv)
{
  shaderc_compilation_result_t result = shaderc_compile_into_spv(
    s_compiler,
    source.c_str(),
    source.size(),
    shaderc_compute_shader,
    filePath.c_str(),
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

bool sgGenerateMainShader(const SgMainShaderParams* params,
                          uint32_t* spvSize,
                          uint32_t** spv,
                          const char** entryPoint)
{
  std::string filePath = s_shadersPath + "/main.comp.hlsl";

  std::string staticSource;
  if (!_sgReadTextFromFile(filePath, staticSource))
  {
    return false;
  }

  std::stringstream ss;
  ss << std::showpoint;
  ss << std::setprecision(std::numeric_limits<float>::digits10);

#define HLSL_TYPE_STRING(ctype) _Generic((ctype), \
  unsigned int: "uint",                           \
  float: "float")

#define APPEND_CONSTANT(name, cvar)     \
  ss << "const ";                       \
  ss << HLSL_TYPE_STRING(params->cvar); \
  ss << " " << name << " = ";           \
  ss << params->cvar << ";\n";

  APPEND_CONSTANT("NUM_THREADS_X", num_threads_x)
  APPEND_CONSTANT("NUM_THREADS_Y", num_threads_y)
  APPEND_CONSTANT("MAX_STACK_SIZE", max_stack_size)
  APPEND_CONSTANT("IMAGE_WIDTH", image_width)
  APPEND_CONSTANT("IMAGE_HEIGHT", image_height)
  APPEND_CONSTANT("SAMPLE_COUNT", spp)
  APPEND_CONSTANT("MAX_BOUNCES", max_bounces)
  APPEND_CONSTANT("CAMERA_ORIGIN_X", camera_position_x)
  APPEND_CONSTANT("CAMERA_ORIGIN_Y", camera_position_y)
  APPEND_CONSTANT("CAMERA_ORIGIN_Z", camera_position_z)
  APPEND_CONSTANT("CAMERA_FORWARD_X", camera_forward_x)
  APPEND_CONSTANT("CAMERA_FORWARD_Y", camera_forward_y)
  APPEND_CONSTANT("CAMERA_FORWARD_Z", camera_forward_z)
  APPEND_CONSTANT("CAMERA_UP_X", camera_up_x)
  APPEND_CONSTANT("CAMERA_UP_Y", camera_up_y)
  APPEND_CONSTANT("CAMERA_UP_Z", camera_up_z)
  APPEND_CONSTANT("CAMERA_VFOV", camera_vfov)
  APPEND_CONSTANT("RR_BOUNCE_OFFSET", rr_bounce_offset)
  APPEND_CONSTANT("RR_INV_MIN_TERM_PROB", rr_inv_min_term_prob)

  std::string source = ss.str() + "\n" + staticSource;
  *entryPoint = "CSMain";

  return _sgCompileHlslToSpv(source, filePath, *entryPoint, spvSize, spv);
}
