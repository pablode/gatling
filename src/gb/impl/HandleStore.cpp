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

#include "HandleStore.h"

#include <assert.h>

namespace gtl
{
  uint64_t GbHandleStore::allocateHandle()
  {
    uint32_t index;
    uint32_t version;

    if (m_freeList.size() > 0)
    {
      index = m_freeList.back();
      m_freeList.pop_back();
      version = m_versions[index];
    }
    else
    {
      index = m_maxIndex++;
      version = 1;

      assert(m_maxIndex < UINT32_MAX);

      m_versions.resize(m_maxIndex);
      m_versions[index] = version;
    }

    return (uint64_t(version) << 32ul) | index;
  }

  bool GbHandleStore::isHandleValid(uint64_t handle) const
  {
    uint32_t index = uint32_t(handle);
    uint32_t version = uint32_t(handle >> 32ul);

    return version > 0 && index <= m_maxIndex && m_versions[index] == version;
  }

  void GbHandleStore::freeHandle(uint64_t handle)
  {
    uint32_t index = uint32_t(handle);

    m_versions[index]++;
    m_freeList.push_back(index);
  }
}
