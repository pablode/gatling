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

#include <gtl/gb/HandleStore.h>
#include <gtl/cgpu/Cgpu.h>

#include "SyncBuffer.h"

namespace gtl
{
  class GgpuStager;

  class GgpuLinearDataStore
  {
  public:
    GgpuLinearDataStore(CgpuDevice device,
                        GgpuStager& stager,
                        uint64_t elementSize,
                        uint32_t minCapacity);

    virtual ~GgpuLinearDataStore();

  public:
    virtual uint64_t allocate();

    virtual void free(uint64_t handle);

    template<typename T>
    T* read(uint64_t handle)
    {
      return (T*) readRaw(handle);
    }

    template<typename T>
    T* write(uint64_t handle)
    {
      return (T*) writeRaw(handle);
    }

    CgpuBuffer buffer() const;

    uint64_t bufferSize() const;

    bool commitChanges();

    uint32_t elementCount() const;

  protected:
    virtual uint8_t* readRaw(uint64_t handle);
    virtual uint8_t* writeRaw(uint64_t handle);
    uint8_t* readFromIndex(uint32_t index);
    uint8_t* writeToIndex(uint32_t index);

  private:
    uint64_t returnOrAllocHandle(uint64_t handle);
    uint64_t returnOrAllocIndex(uint32_t index);

  private:
    CgpuDevice m_device;
    uint64_t m_elementSize;
    uint32_t m_minCapacity;
    uint32_t m_elementCount;

    GbHandleStore m_handleStore;
    GgpuSyncBuffer m_buffer;
    uint8_t* m_mappedMem = nullptr;
  };
}
