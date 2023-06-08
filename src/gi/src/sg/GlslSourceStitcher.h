//
// Copyright (C) 2023 Pablo Delgado Krämer
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

#include <stdint.h>
#include <string_view>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace gi::sg
{
  class GlslSourceStitcher
  {
  public:
    GlslSourceStitcher();

    void appendVersion();

    void appendDefine(std::string_view name);
    void appendDefine(std::string_view name, int32_t value);
    void appendDefine(std::string_view name, float value);

    void appendRequiredExtension(std::string_view name);

    void appendString(std::string_view value);

    bool appendSourceFile(fs::path path);

    bool replaceFirst(std::string_view substring, std::string_view replacement);

    std::string source();

  private:
    std::stringstream m_source;
  };
}
