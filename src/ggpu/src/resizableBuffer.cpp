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

#include "resizableBuffer.h"
#include "fencedCallbackExecutor.h"

#include <assert.h>

namespace gtl
{
  GgpuResizableBuffer::GgpuResizableBuffer(CgpuDevice device,
                                           GgpuFencedCallbackExecutor& fencedCallbackExecutor,
                                           CgpuBufferUsageFlags usageFlags,
                                           CgpuMemoryPropertyFlags memoryProperties)
    : m_device(device)
    , m_fencedCallbackExecutor(fencedCallbackExecutor)
    , m_usageFlags(usageFlags)
    , m_memoryProperties(memoryProperties)
  {
  }

  GgpuResizableBuffer::~GgpuResizableBuffer()
  {
    // Buffer has to be destroyed with resize(0) before destruction.
    assert(m_size == 0);
  }

  CgpuBuffer GgpuResizableBuffer::buffer() const
  {
    return m_buffer;
  }

  uint64_t GgpuResizableBuffer::size() const
  {
    return m_size;
  }

  bool GgpuResizableBuffer::resize(CgpuCommandBuffer commandBuffer, uint64_t newSize)
  {
    if (newSize == m_size)
    {
      return true;
    }

    if (newSize == 0)
    {
      if (m_buffer.handle)
      {
        cgpuDestroyBuffer(m_device, m_buffer);
        m_buffer.handle = 0;
      }

      m_size = 0;
      return true;
    }

    // Create new, larger buffer.
    CgpuBuffer buffer;
    if (!cgpuCreateBuffer(m_device, m_usageFlags, m_memoryProperties, newSize, &buffer))
    {
      return false;
    }

    // Copy old buffer data if needed.
    bool copySuccess = (m_size == 0) || cgpuCmdCopyBuffer(commandBuffer, m_buffer, 0, buffer, 0, m_size);

    m_size = newSize;

    // Swap buffers, so that we always destroy the unused one.
    {
      CgpuBuffer temp = m_buffer;
      m_buffer = buffer;
      buffer = temp;
    }

    // Delete old buffer once unused.
    if (buffer.handle)
    {
      m_fencedCallbackExecutor.enqueueCallbackExecution(commandBuffer, [buffer](CgpuDevice device) {
        cgpuDestroyBuffer(device, buffer);
      });
    }

    return copySuccess;
  }
}
