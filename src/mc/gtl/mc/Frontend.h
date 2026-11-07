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

#include <memory>
#include <string>
#include <string_view>

#include "Material.h"
#include "MaterialParameters.h"
#include "MtlxDocPatch.h"

namespace gtl
{
  struct McMaterial;
  class McMdlMaterialCompiler;
  class McMtlxMdlCodeGen;
  class McRuntime;

  class McFrontend
  {
  public:
    McFrontend(const MaterialX::DocumentPtr mtlxStdLib,
               const std::string& customMtlxNodesPath,
               McRuntime& mdlRuntime);

  public:
    MaterialX::DocumentPtr parseMtlxDoc(std::string_view mtlxSrc);

    McMtlxMaterialDoc patchMtlxDoc(const MaterialX::DocumentPtr& doc,
                                   const char* materialName,
                                   bool usdPreviewSurfacePresenceOverride,
                                   uint32_t maxParameterCount);

    McMaterial* createFromMtlxDoc(const MaterialX::DocumentPtr& doc,
                                  const MaterialX::NodePtr& surfaceShader,
                                  McOpacityClassification opacityClass);

    McMaterial* createFromMdlFile(const char* filePath, std::string_view subIdentifier, const McMaterialParameters& parameters = {});

  private:
    McMaterial* createFromMdlStr(std::string_view mdlSrc,
                                 std::string_view subIdentifier,
                                 McOpacityClassification opacityClass,
                                 bool isAnimated);

  private:
    MaterialX::DocumentPtr m_mtlxStdLib;
    std::shared_ptr<McMdlMaterialCompiler> m_mdlMaterialCompiler;
    std::shared_ptr<McMtlxMdlCodeGen> m_mtlxMdlCodeGen;
    std::shared_ptr<McMtlxDocumentPatcher> m_mtlxDocPatcher;
  };
}
