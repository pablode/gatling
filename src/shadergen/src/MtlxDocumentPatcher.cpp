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

#include "MtlxDocumentPatcher.h"

#include <MaterialXCore/Types.h>

#include <assert.h>

namespace mx = MaterialX;

void _SanitizeFilePath(std::string& path)
{
  // The MDL SDK does not take raw OS paths. First, only forward-facing slashes are allowed.
  std::replace(path.begin(), path.end(), '\\', '/');

  // Second, only UNIX-style absolute paths ('/' prefix, no double colon) are valid.
  bool hasDriveSpecifier = path.size() >= 2 && path[1] == ':';

  if (hasDriveSpecifier)
  {
    path[1] = path[0];
    path[0] = '/';
  }
}

void _SanitizeFilePaths(MaterialX::DocumentPtr document)
{
  for (auto treeIt = document->traverseTree(); treeIt != mx::TreeIterator::end(); ++treeIt)
  {
    mx::ElementPtr elem = treeIt.getElement();

    mx::PortElementPtr portElem = elem->asA<mx::PortElement>();
    if (!portElem)
    {
      continue;
    }

    std::string portType = portElem->getType();
    if (portType != mx::FILENAME_TYPE_STRING)
    {
      continue;
    }

    mx::ValuePtr valuePtr = portElem->getValue();
    if (!valuePtr)
    {
      continue;
    }

    std::string path = valuePtr->asA<std::string>();

    _SanitizeFilePath(path);

    portElem->setValue(path, mx::FILENAME_TYPE_STRING);
  }
}

namespace sg
{
  void MtlxDocumentPatcher::patch(MaterialX::DocumentPtr document)
  {
    _SanitizeFilePaths(document);
  }
}
