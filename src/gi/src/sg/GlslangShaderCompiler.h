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

#pragma once

#include <vector>
#include <string_view>
#include <filesystem>

namespace fs = std::filesystem;

namespace gi::sg
{
  class GlslangShaderCompiler
  {
  public:
    enum class ShaderStage
    {
      AnyHit,
      ClosestHit,
      Compute,
      Miss,
      RayGen
    };

  public:
    GlslangShaderCompiler(const fs::path& shaderPath);

  public:
    bool compileGlslToSpv(ShaderStage stage,
                          std::string_view source,
                          std::vector<uint8_t>& spv);

    static bool init();

    static void deinit();

  private:
    std::shared_ptr<class _FileIncluder> m_fileIncluder;
  };
}
