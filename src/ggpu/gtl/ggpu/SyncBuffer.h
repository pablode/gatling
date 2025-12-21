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
#include <memory>

#include <gtl/cgpu/Cgpu.h>

#include "ResizableBuffer.h"

namespace gtl
{
  class GgpuStager;
  class GgpuDeleteQueue;

  class GgpuSyncBuffer
  {
  public:
    GgpuSyncBuffer(CgpuContext* ctx,
                   GgpuStager& stager,
                   GgpuDeleteQueue& deleteQueue,
                   uint64_t elementSize,
                   CgpuBufferUsage bufferUsage = CgpuBufferUsage::Storage);

    GgpuSyncBuffer(const GgpuSyncBuffer&) = delete;
    GgpuSyncBuffer& operator=(const GgpuSyncBuffer&) = delete;

    ~GgpuSyncBuffer();

  public:
    uint8_t* read(uint64_t byteOffset, uint64_t byteSize);

    uint8_t* write(uint64_t byteOffset, uint64_t byteSize);

    template<typename T>
    T* read(uint64_t offset, uint64_t range)
    {
      return (T*) write(offset * m_elementSize, range * m_elementSize);
    }

    template<typename T>
    T* write(uint64_t offset, uint64_t range)
    {
      return (T*) write(offset * m_elementSize, range * m_elementSize);
    }

    bool resize(CgpuContext* ctx, CgpuCommandBuffer commandBuffer, uint64_t newSize);

    CgpuBuffer buffer() const;

    uint64_t byteSize() const;

    bool commitChanges();

  private:
    CgpuContext* m_ctx;
    GgpuStager& m_stager;
    uint64_t m_elementSize;

    uint64_t m_size = 0;
    GgpuResizableBuffer m_deviceBuffer;
    std::unique_ptr<uint8_t[]> m_hostMem;

    // TODO: list of ranges or an interval tree for fine granular tracking
    uint64_t m_dirtyRangeBegin = UINT64_MAX;
    uint64_t m_dirtyRangeEnd = 0;
  };
}
