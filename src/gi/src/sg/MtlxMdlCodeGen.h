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

namespace gi::sg
{
  class MtlxMdlCodeGen
  {
  public:
    explicit MtlxMdlCodeGen(const char* mtlxLibPath);

  public:
    bool translate(std::string_view mtlxSrc, std::string& mdlSrc, std::string& subIdentifier, bool& isOpaque);

  private:
    const MaterialX::FileSearchPath m_mtlxLibPath;
    MaterialX::DocumentPtr m_stdLib;
    MaterialX::ShaderGeneratorPtr m_shaderGen;
  };
}
