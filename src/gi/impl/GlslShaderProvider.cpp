//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#include "GlslShaderProvider.h"

#include <gtl/ggpu/DelayedResourceDestroyer.h>
#include <gtl/gb/Log.h>

#include <sstream>
#include <vector>

namespace detail
{
  bool readTextFromFile(const fs::path& filePath, std::string& text)
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
}

namespace gtl
{
  GiGlslShaderProvider::GiGlslShaderProvider(const fs::path& shaderDir,
                                             CgpuDevice device,
                                             GgpuDelayedResourceDestroyer& delayedResourceDestroyer)
    : m_shaderDir(shaderDir)
    , m_device(device)
    , m_delayedResourceDestroyer(delayedResourceDestroyer)
    , m_compiler(shaderDir)
  {
  }

  GiGlslShaderProvider::~GiGlslShaderProvider()
  {
    for (const auto& kvPair : m_cache)
    {
      CgpuShader shader = kvPair.second;

      m_delayedResourceDestroyer.enqueueDestruction(shader);
    }
  }

  CgpuShader GiGlslShaderProvider::provide(GiShaderStage stage,
                                           const char* fileName,
                                           GiGlslDefines* glslDefines,
                                           GiGlslSourceTransformer sourceTransformer)
  {
    std::string fileSource;
    fs::path shaderPath = m_shaderDir / fileName;
    if (!detail::readTextFromFile(shaderPath, fileSource))
    {
      GB_ERROR("failed to read {}", shaderPath);
      return CgpuShader{0};
    }

    std::string preamble;
    if (glslDefines)
    {
      for (const auto& kv : glslDefines->map)
      {
         GB_FMT_TO(std::back_inserter(preamble), "#define {} {}\n", kv.first, kv.second);
      }
    }

    std::string source = preamble + fileSource;
    if (sourceTransformer)
    {
      source = sourceTransformer(source.data());
    }

    std::hash<std::string> hasher;
    size_t hash = hasher(source);
    GB_DEBUG("shader {} ({}) has hash {}", fileName, int(stage), hash);

    auto cacheIt = m_cache.find(hash);
    if (cacheIt != m_cache.end())
    {
      GB_DEBUG("entry found in cache");
      return cacheIt->second;
    }

    std::vector<uint8_t> spv;
    if (!m_compiler.compileGlslToSpv(stage, source, spv))
    {
      return CgpuShader{0};
    }

    CgpuShader shader;
    if (!cgpuCreateShader(m_device, {
                            .size = spv.size(),
                            .source = spv.data(),
                            .stageFlags = (CgpuShaderStageFlags) stage,
                          }, &shader))
    {
      return CgpuShader{0};
    }

    m_cache[hash] = shader;
    return shader;
  }
}
