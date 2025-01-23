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

namespace mx = MaterialX;

namespace gtl
{
  McMtlxDocumentParser::McMtlxDocumentParser(const MaterialX::DocumentPtr stdLib)
    : m_stdLib(stdLib)
  {
  }

  MaterialX::DocumentPtr McMtlxDocumentParser::parse(std::string_view str)
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
}
