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

namespace gtl
{
  class GiLinearDataStoreGpu
  {
  public:
    GiLinearDataStoreGpu(CgpuDevice device, uint64_t elementSize, uint32_t initialCapacity);

    ~GiLinearDataStoreGpu();

  public:
    uint64_t allocate();

    void free(uint64_t handle);

    bool get(uint64_t handle, void** element);

    template<typename T>
    bool get(uint64_t handle, T** element)
    {
      return get(handle, (void**)element);
    }

    CgpuBuffer buffer() const;

    uint64_t bufferSize() const;

  private:
    bool resizeBuffer(uint64_t newSize);

  private:
    CgpuDevice m_device;
    uint64_t m_elementSize;

    GbHandleStore m_handleStore;
    CgpuBuffer m_buffer = { CGPU_INVALID_HANDLE };
    uint64_t m_bufferSize = 0;
    uint8_t* m_mappedMem = nullptr;
  };
}
