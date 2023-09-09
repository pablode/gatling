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

#include "GlslStitcher.h"

#include <string>
#include <fstream>

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
  GiGlslStitcher::GiGlslStitcher()
  {
    // Full float precision so that we don't cut off epsilons
    m_source.precision(std::numeric_limits<float>::max_digits10);
    // Ensure no integer literals are emitted for floats
    m_source.setf(std::ios::fixed | std::ios::showpoint);
  }

  void GiGlslStitcher::appendVersion()
  {
    m_source << "#version 460 core\n";
  }

  void GiGlslStitcher::appendDefine(std::string_view name)
  {
    m_source << "#define " << name << "\n";
  }

  void GiGlslStitcher::appendDefine(std::string_view name, int32_t value)
  {
    m_source << "#define " << name << " " << value << "\n";
  }

  void GiGlslStitcher::appendDefine(std::string_view name, float value)
  {
    m_source << "#define " << name << " " << value << "\n";
  }

  void GiGlslStitcher::appendRequiredExtension(std::string_view name)
  {
    m_source << "#extension " << name << ": require\n";
  }

  void GiGlslStitcher::appendString(std::string_view value)
  {
    m_source << value;
  }

  bool GiGlslStitcher::appendSourceFile(fs::path path)
  {
    std::string text;
    if (!detail::readTextFromFile(path, text))
    {
      return false;
    }

    appendString(text);
    return true;
  }

  bool GiGlslStitcher::replaceFirst(std::string_view substring, std::string_view replacement)
  {
    std::string tmp = m_source.str();

    size_t location = tmp.find(substring);
    if (location == std::string::npos)
    {
      return false;
    }

    tmp.replace(location, substring.length(), replacement);

    m_source = std::stringstream(tmp);
    return true;
  }

  std::string GiGlslStitcher::source()
  {
    return m_source.str();
  }
}
