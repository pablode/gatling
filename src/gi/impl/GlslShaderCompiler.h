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

namespace gtl
{
// TODO: this sucks! or add static_assert for size
  enum class GiShaderStage
  {
    Compute = 0x020,
    RayGen = 0x100,
    AnyHit = 0x200,
    ClosestHit = 0x400,
    Miss = 0x800
  };

  class GiGlslShaderCompiler
  {
  public:
    GiGlslShaderCompiler(const fs::path& shaderPath);

    ~GiGlslShaderCompiler();

  public:
    bool compileGlslToSpv(GiShaderStage stage,
                          std::string_view source,
                          std::vector<uint8_t>& spv);

  private:
    std::shared_ptr<class _FileIncluder> m_fileIncluder;
  };
}
