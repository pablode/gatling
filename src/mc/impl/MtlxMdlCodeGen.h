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

#include <stdint.h>
#include <string>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXGenShader/ShaderGenerator.h>

namespace gtl
{
  class McMtlxMdlCodeGen
  {
  public:
    explicit McMtlxMdlCodeGen(const MaterialX::DocumentPtr mtlxStdLib);

  public:
    bool translate(MaterialX::DocumentPtr mtlxDoc, std::string& mdlSrc, std::string& subIdentifier, bool& hasCutoutTransparency);
    bool translate(std::string_view mtlxStr, std::string& mdlSrc, std::string& subIdentifier, bool& hasCutoutTransparency);

  private:
    MaterialX::ShaderGeneratorPtr m_shaderGen;
    MaterialX::DocumentPtr m_baseDoc;
  };
}
