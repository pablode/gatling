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

#include "ResizableBuffer.h"

#include "DelayedResourceDestroyer.h"

namespace gtl
{
  GgpuResizableBuffer::GgpuResizableBuffer(CgpuDevice device,
                                           GgpuDelayedResourceDestroyer& delayedResourceDestroyer,
                                           CgpuBufferUsageFlags usageFlags,
                                           CgpuMemoryPropertyFlags memoryProperties)
    : m_device(device)
    , m_delayedResourceDestroyer(delayedResourceDestroyer)
    , m_usageFlags(usageFlags)
    , m_memoryProperties(memoryProperties)
  {
  }

  GgpuResizableBuffer::~GgpuResizableBuffer()
  {
    if (m_buffer.handle)
    {
      m_delayedResourceDestroyer.enqueueDestruction(m_buffer);
    }
  }

  CgpuBuffer GgpuResizableBuffer::buffer() const
  {
    return m_buffer;
  }

  uint64_t GgpuResizableBuffer::size() const
  {
    return m_size;
  }

  bool GgpuResizableBuffer::resize(uint64_t newSize, CgpuCommandBuffer commandBuffer)
  {
    if (newSize == m_size)
    {
      return true;
    }

    if (newSize == 0)
    {
      if (m_buffer.handle)
      {
        m_delayedResourceDestroyer.enqueueDestruction(m_buffer);
        m_buffer.handle = 0;
      }

      m_size = 0;
      return true;
    }

    // Create new, larger buffer.
    bool result = false;

    CgpuBuffer buffer;
    if (!cgpuCreateBuffer(m_device, {
                            .usage = m_usageFlags,
                            .memoryProperties = m_memoryProperties,
                            .size = newSize,
                            .debugName = "[resizable buffer]"
                          }, &buffer))
    {
      goto cleanup;
    }

    // Copy old buffer data if needed.
    if (m_size > 0)
    {
      if (!cgpuCmdCopyBuffer(commandBuffer, m_buffer, 0, buffer, 0, m_size))
        goto cleanup;
    }

    // Swap buffers, so that we always destroy the unused one.
    {
      CgpuBuffer temp = m_buffer;
      m_buffer = buffer;
      buffer = temp;
    }

    m_size = newSize;

    result = true;

  cleanup:
    if (buffer.handle)
    {
      m_delayedResourceDestroyer.enqueueDestruction(buffer);
    }

    return result;
  }
}
