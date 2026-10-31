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

#include "DenseDataStore.h"

#include <string.h>

namespace gtl
{
  GgpuDenseDataStore::GgpuDenseDataStore(CgpuContext* ctx,
                                         GgpuStager& stager,
                                         GgpuDeleteQueue& deleteQueue,
                                         uint64_t elementSize,
                                         uint32_t minCapacity)
    : GgpuLinearDataStore(ctx, stager, deleteQueue, elementSize, minCapacity)
    , m_elementSize(elementSize)
  {
    m_handleSlots.reserve(minCapacity);
    m_indexHandles.reserve(minCapacity);
  }

  uint64_t GgpuDenseDataStore::allocate()
  {
    uint64_t handle = GgpuLinearDataStore::allocate();

    uint32_t handleIndex = uint32_t(handle);
    if (handleIndex >= m_handleSlots.size())
    {
      m_handleSlots.resize(handleIndex + 1);
    }

    m_handleSlots[handleIndex] = { m_highestIndex, uint32_t(handle >> 32) };
    m_indexHandles.push_back(handleIndex);
    m_highestIndex += 1;

    return handle;
  }

  void GgpuDenseDataStore::free(uint64_t handle)
  {
    uint32_t handleIndex = uint32_t(handle);
    if (handleIndex >= m_handleSlots.size())
    {
      assert(false);
      return;
    }

    HandleSlot& slot = m_handleSlots[handleIndex];
    if (slot.version != uint32_t(handle >> 32) || slot.index == UINT32_MAX)
    {
      assert(false);
      return;
    }

    uint32_t freedIndex = slot.index;
    uint32_t lastIndex = m_highestIndex - 1;

    // Move the last element into the freed slot to keep the storage dense.
    if (freedIndex != lastIndex)
    {
      uint32_t lastHandleIndex = m_indexHandles[lastIndex];
      HandleSlot& lastSlot = m_handleSlots[lastHandleIndex];
      assert(lastSlot.index == lastIndex);

      uint8_t* dstPtr = GgpuLinearDataStore::writeToIndex(freedIndex);
      uint8_t* srcPtr = GgpuLinearDataStore::readFromIndex(lastIndex);
      memcpy(dstPtr, srcPtr, m_elementSize);

      lastSlot.index = freedIndex;
      m_indexHandles[freedIndex] = lastHandleIndex;
    }

    slot.index = UINT32_MAX;
    slot.version = 0;
    m_indexHandles.pop_back();
    m_highestIndex--;

    GgpuLinearDataStore::free(handle);
  }

  uint8_t* GgpuDenseDataStore::readRaw(uint64_t handle)
  {
    uint32_t handleIndex = uint32_t(handle);
    if (handleIndex >= m_handleSlots.size())
    {
      assert(false);
      return nullptr;
    }

    const HandleSlot& slot = m_handleSlots[handleIndex];
    if (slot.version != uint32_t(handle >> 32) || slot.index == UINT32_MAX)
    {
      assert(false);
      return nullptr;
    }

    return GgpuLinearDataStore::readFromIndex(slot.index);
  }

  uint8_t* GgpuDenseDataStore::writeRaw(uint64_t handle)
  {
    uint32_t handleIndex = uint32_t(handle);
    if (handleIndex >= m_handleSlots.size())
    {
      assert(false);
      return nullptr;
    }

    const HandleSlot& slot = m_handleSlots[handleIndex];
    if (slot.version != uint32_t(handle >> 32) || slot.index == UINT32_MAX)
    {
      assert(false);
      return nullptr;
    }

    return GgpuLinearDataStore::writeToIndex(slot.index);
  }
}
