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

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include <SmallVector.h>
#include <HandleStore.h>

namespace gtl
{
  template<typename T, uint32_t C>
  class GbLinearDataStore
  {
  public:
    uint64_t allocate()
    {
      return m_handleStore.allocateHandle();
    }

    void free(uint64_t handle)
    {
      assert(m_handleStore.isHandleValid(handle));

      m_handleStore.freeHandle(handle);
    }

    bool get(uint64_t handle, T** object)
    {
      if (!m_handleStore.isHandleValid(handle))
      {
        assert(false);
        return false;
      }

      uint32_t index = uint32_t(handle);
      if (index >= m_objects.size())
      {
        m_objects.resize(index + 1);
      }

      *object = &m_objects[index];
      return true;
    }

  private:
    GbHandleStore m_handleStore;
    GbSmallVector<T, C> m_objects;
  };
}
