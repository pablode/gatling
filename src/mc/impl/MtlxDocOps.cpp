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

#include <gtl/gb/Log.h>

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

  mx::NodePtr McMtlxFindSurfaceShader(const mx::DocumentPtr& doc, const char* name)
  {
    // Find material in document
    auto materialNodes = doc->getMaterialNodes();

    if (materialNodes.size() != 1)
    {
      // HdMtlx generates a new document for each material; there should be a single material
      GB_ERROR("unexpected material count in mtlx document for material {}", name ? name : "<unnamed>");
      return nullptr;
    }

    mx::NodePtr materialNode = materialNodes[0];

    // Extract surface shader node
    auto surfaceShaderNodes = mx::getShaderNodes(materialNode, mx::SURFACE_SHADER_TYPE_STRING);

    mx::TypedElementPtr renderableElement = materialNode;
    if (!surfaceShaderNodes.empty())
    {
      renderableElement = *surfaceShaderNodes.begin();

      if (surfaceShaderNodes.size() > 1)
      {
        GB_WARN("found multiple surface shaders for material {}; choosing first", name ? name : "<unnamed>");
      }
    }

    mx::ElementPtr surfaceElement = doc->getDescendant(renderableElement->getNamePath());
    if (!surfaceElement)
    {
      GB_WARN("could not find surface shader for material {}", name ? name : "<unnamed>");
      return nullptr;
    }

    GB_DEBUG("found surface shader: {}", surfaceElement->getName());
    return surfaceElement->asA<mx::Node>();
  }
}
