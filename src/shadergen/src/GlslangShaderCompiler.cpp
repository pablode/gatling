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

#include <shaderc/env.h>
#include <libshaderc_util/file_finder.h>
#include <libshaderc_util/io_shaderc.h>

#include <fstream>

namespace detail
{
  struct IncludeStringKeeper
  {
    std::string sourcePath;
    std::vector<char> content;
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

    std::vector<char> content;
    if (!shaderc_util::ReadFile(sourcePath, &content))
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
    result->content = keeper->content.data();
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

    shaderc_compile_options_set_source_language(m_compileOptions, shaderc_source_language_glsl);
    shaderc_compile_options_set_optimization_level(m_compileOptions, shaderc_optimization_level_performance);
    shaderc_compile_options_set_target_env(m_compileOptions, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    shaderc_compile_options_set_target_spirv(m_compileOptions, shaderc_spirv_version_1_3);

#ifdef NDEBUG
    shaderc_compile_options_set_suppress_warnings(m_compileOptions);
#else
    shaderc_compile_options_set_generate_debug_info(m_compileOptions);
    shaderc_compile_options_set_warnings_as_errors(m_compileOptions);
#endif

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

  bool GlslangShaderCompiler::compileGlslToSpv(std::string_view source,
                                               std::string_view filePath,
                                               std::string_view entryPoint,
                                               std::vector<uint8_t>& spv)
  {
    shaderc_compilation_result_t result = shaderc_compile_into_spv(
      m_compiler,
      source.data(),
      source.size(),
      shaderc_compute_shader,
      filePath.data(),
      entryPoint.data(),
      m_compileOptions
    );

    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success)
    {
      fprintf(stderr, "Failed to compile shader: %s\n", shaderc_result_get_error_message(result));
      shaderc_result_release(result);
      return false;
    }

    size_t spvSize = shaderc_result_get_length(result);
    spv.resize(spvSize);

    const char* data = shaderc_result_get_bytes(result);
    memcpy(spv.data(), data, spvSize);

    shaderc_result_release(result);
    return true;
  }
}
