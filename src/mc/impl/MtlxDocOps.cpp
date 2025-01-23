//
// Copyright (C) 2025 Pablo Delgado Krämer
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

#include "MtlxDocOps.h"

#include <MaterialXFormat/Util.h>
#include <MaterialXGenShader/Util.h>

namespace mx = MaterialX;

namespace gtl
{
  McMtlxDocumentParser::McMtlxDocumentParser(const mx::DocumentPtr& stdLib)
    : m_stdLib(stdLib)
  {
  }

  mx::DocumentPtr McMtlxDocumentParser::parse(std::string_view str)
  {
    try
    {
      mx::DocumentPtr doc = mx::createDocument();
      doc->importLibrary(m_stdLib);
      mx::readFromXmlString(doc, str.data());
      return doc;
    }
    catch (const std::exception& ex)
    {
      return nullptr;;
    }
  }

  mx::NodePtr McMtlxFindSurfaceShader(const mx::DocumentPtr& doc)
  {
    // Find renderable element.
    std::vector<mx::TypedElementPtr> renderableElements;
#if (MATERIALX_MAJOR_VERSION > 1) || \
    (MATERIALX_MAJOR_VERSION == 1 && MATERIALX_MINOR_VERSION > 38) || \
    (MATERIALX_MAJOR_VERSION == 1 && MATERIALX_MINOR_VERSION == 38 && MATERIALX_BUILD_VERSION > 7)
    renderableElements = mx::findRenderableElements(doc);
#else
    mx::findRenderableElements(doc, renderableElements);
#endif

    for (mx::TypedElementPtr elem : renderableElements)
    {
      // Extract surface shader node.
      mx::TypedElementPtr renderableElement = renderableElements.at(0);
      mx::NodePtr node = renderableElement->asA<mx::Node>();

      if (node && node->getType() == mx::MATERIAL_TYPE_STRING)
      {
        auto shaderNodes = mx::getShaderNodes(node, mx::SURFACE_SHADER_TYPE_STRING);
        if (!shaderNodes.empty())
        {
          renderableElement = *shaderNodes.begin();
        }
      }

      mx::ElementPtr surfaceElement = doc->getDescendant(renderableElement->getNamePath());
      if (!surfaceElement)
      {
        return nullptr;
      }

      return surfaceElement->asA<mx::Node>();
    }

    return nullptr;
  }
}
