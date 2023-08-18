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

#include "denseDataStore.h"

#include <string.h>

namespace gtl
{
  GgpuDenseDataStore::GgpuDenseDataStore(CgpuDevice device,
                                         GgpuStager& stager,
                                         uint64_t elementSize,
                                         uint32_t minCapacity)
    : GgpuLinearDataStore(device, stager, elementSize, minCapacity)
    , m_elementSize(elementSize)
  {
    m_indexMap.reserve(minCapacity);
  }

  uint64_t GgpuDenseDataStore::allocate()
  {
    uint64_t handle = GgpuLinearDataStore::allocate();

    m_indexMap[handle] = m_highestIndex;
    m_highestIndex += 1;

    return handle;
  }

  void GgpuDenseDataStore::free(uint64_t handle)
  {
    if (m_highestIndex == 1)
    {
      // No swapping needed/possible.
      m_highestIndex = 0;
      return;
    }

    auto indexIt = m_indexMap.find(handle);
    if (indexIt == m_indexMap.end())
    {
      assert(false);
      return;
    }

    uint32_t freedIndex = indexIt->second;
    uint8_t* dstPtr = GgpuLinearDataStore::writeToIndex(freedIndex);
    uint8_t* srcPtr = GgpuLinearDataStore::readFromIndex(m_highestIndex);

    memcpy((void*) dstPtr, (void*) srcPtr, m_elementSize);
    m_highestIndex--;

    GgpuLinearDataStore::free(handle);
  }

  uint8_t* GgpuDenseDataStore::readRaw(uint64_t handle)
  {
    auto indexIt = m_indexMap.find(handle);
    if (indexIt == m_indexMap.end())
    {
      assert(false);
      return nullptr;
    }

    return GgpuLinearDataStore::readFromIndex(indexIt->second);
  }

  uint8_t* GgpuDenseDataStore::writeRaw(uint64_t handle)
  {
    auto indexIt = m_indexMap.find(handle);
    if (indexIt == m_indexMap.end())
    {
      assert(false);
      return nullptr;
    }

    return GgpuLinearDataStore::writeToIndex(indexIt->second);
  }
}
