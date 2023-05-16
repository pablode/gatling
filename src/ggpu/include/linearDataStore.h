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

#include <cgpu.h>
#include <handleStore.h>

#include "syncBuffer.h"

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

    ~GgpuLinearDataStore();

  public:
    uint64_t allocate();

    void free(uint64_t handle);

    template<typename T>
    T* getForReading(uint64_t handle)
    {
      return (T*) getForReadingRaw(handle);
    }

    template<typename T>
    T* getForWriting(uint64_t handle)
    {
      return (T*) getForWritingRaw(handle);
    }

    CgpuBuffer buffer() const;

    uint64_t bufferSize() const;

  private:
    uint64_t resolveOffsetAndAlloc(uint64_t handle);

    uint8_t* getForReadingRaw(uint64_t handle);

    uint8_t* getForWritingRaw(uint64_t handle);

  private:
    CgpuDevice m_device;
    uint64_t m_elementSize;
    uint32_t m_minCapacity;

    GbHandleStore m_handleStore;
    GgpuSyncBuffer m_buffer;
    uint8_t* m_mappedMem = nullptr;
  };
}
