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

#include "Runtime.h"

#include "MdlRuntime.h"

namespace gtl
{
  McRuntime::McRuntime(McMdlRuntime* mdlRuntime)
    : m_mdlRuntime(mdlRuntime)
  {
  }

  McRuntime::~McRuntime()
  {
    delete m_mdlRuntime;
  }

  McMdlRuntime& McRuntime::getMdlRuntime() const
  {
    return *m_mdlRuntime;
  }

  McRuntime* McLoadRuntime(std::string_view libDir, const std::vector<std::string>& mdlSearchPaths)
  {
    McMdlRuntime* r = new McMdlRuntime();
    if (!r->init(libDir, mdlSearchPaths))
    {
        delete r;
        return nullptr;
    }
    return new McRuntime(r);
  }
}
