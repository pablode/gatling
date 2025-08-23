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

#include <MaterialXCore/Document.h>

#include <string>
#include <vector>

#include "MaterialParameters.h"

namespace gtl
{
  struct McMaterial;
  class McMdlMaterialCompiler;
  class McRuntime;
  class McMtlxMdlCodeGen;

  class McFrontend
  {
  public:
    McFrontend(const MaterialX::DocumentPtr mtlxStdLib,
               const std::string& customMtlxNodesPath,
               McRuntime& mdlRuntime);

  public:
    McMaterial* createFromMtlxStr(std::string_view docStr);

    McMaterial* createFromMtlxDoc(const MaterialX::DocumentPtr doc);

    McMaterial* createFromMdlFile(const char* filePath, std::string_view subIdentifier, const McMaterialParameters& parameters = {});

  private:
    McMaterial* createFromMdlStr(std::string_view mdlSrc, std::string_view subIdentifier, bool isOpaque);

  private:
    std::shared_ptr<McMdlMaterialCompiler> m_mdlMaterialCompiler;
    std::shared_ptr<McMtlxMdlCodeGen> m_mtlxMdlCodeGen;
  };
}
