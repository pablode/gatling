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

#pragma once

#include <functional>
#include <filesystem>
#include <unordered_map>

#include <gtl/gb/Fmt.h>
#include <gtl/cgpu/Cgpu.h>

#include "GlslShaderCompiler.h"

namespace fs = std::filesystem;

namespace gtl
{
  class GgpuDelayedResourceDestroyer;

  struct GiGlslDefines
  {
    std::unordered_map<const char*, std::string> map;

    template<typename T>
    void setDefine(const char* name, const T& value)
    {
      map[name] = GB_FMT("{}", value);
    }

    void setDefine(const char* name)
    {
      setDefine(name, "");
    }

    void setConditionalDefine(bool cond, const char* name)
    {
      if (cond)
      {
        setDefine(name);
      }
    }
  };

  using GiGlslSourceTransformer = std::function<std::string(const char*)>;

  class GiGlslShaderProvider
  {
  public:
    GiGlslShaderProvider(const fs::path& shaderDir,
                         CgpuDevice device,
                         GgpuDelayedResourceDestroyer& delayedResourceDestroyer);

    ~GiGlslShaderProvider();

  public:
    CgpuShader provide(GiShaderStage stage,
                       const char* fileName,
                       GiGlslDefines* glslDefines = nullptr,
                       GiGlslSourceTransformer sourceTransformer = nullptr);

  private:
    const fs::path& m_shaderDir;
    CgpuDevice m_device;
    GgpuDelayedResourceDestroyer& m_delayedResourceDestroyer;
    GiGlslShaderCompiler m_compiler;

    std::unordered_map<size_t/*hash*/, CgpuShader> m_cache;
  };
}
