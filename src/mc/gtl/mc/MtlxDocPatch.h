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

#ifdef OCIO
#include <OpenColorIO/OpenColorIO.h>
namespace ocio = OCIO_NAMESPACE;
#endif

#include <stdint.h>
#include <vector>

#include "Material.h"

namespace gtl
{
  // A MaterialX document which has been patched for Gatling's requirements and parameterized for
  // material deduplication, together with the derived properties needed to compile it and to
  // build its GPU parameter buffer.
  struct McMtlxMaterialDoc
  {
    MaterialX::DocumentPtr doc;
    MaterialX::NodePtr surfaceShader;
    McOpacityClassification opacityClass;
    std::vector<McMtlxNetworkParameter> parameters;
  };

  class McMtlxDocumentPatcher
  {
  public:
    McMtlxDocumentPatcher(const MaterialX::DocumentPtr mtlxStdLib, const std::string& customNodesPath);

    McMtlxMaterialDoc patch(const MaterialX::DocumentPtr& document,
                            const char* materialName,
                            bool usdPreviewSurfacePresenceOverride,
                            uint32_t maxParameterCount);

  private:
    MaterialX::DocumentPtr m_customNodesDoc;
#ifdef OCIO
    ocio::ConstConfigRcPtr m_ocioConfig;
#endif
  };
}
