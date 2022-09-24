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

namespace sg
{
  class GlslangShaderCompiler
  {
  public:
    GlslangShaderCompiler(const std::string& shaderPath);

    ~GlslangShaderCompiler();

  public:
    bool compileGlslToSpv(std::string_view source,
                          std::string_view filePath,
                          std::vector<uint8_t>& spv);

    static bool init();

    static void deinit();

  private:
    void* m_fileIncluder;
  };
}
