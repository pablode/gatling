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

namespace gtl
{
  GgpuResizableBuffer::GgpuResizableBuffer(CgpuDevice device,
                                           CgpuBufferUsageFlags usageFlags,
                                           CgpuMemoryPropertyFlags memoryProperties)
    : m_device(device)
    , m_usageFlags(usageFlags)
    , m_memoryProperties(memoryProperties)
  {
  }

  GgpuResizableBuffer::~GgpuResizableBuffer()
  {
    // TODO: add to deletion queue instead
    if (m_buffer.handle != CGPU_INVALID_HANDLE)
      cgpuDestroyBuffer(m_device, m_buffer);
  }

  CgpuBuffer GgpuResizableBuffer::buffer() const
  {
    return m_buffer;
  }

  uint64_t GgpuResizableBuffer::size() const
  {
    return m_size;
  }

  bool GgpuResizableBuffer::resize(uint64_t newSize)
  {
    if (newSize == m_size)
    {
      return true;
    }

    if (newSize == 0)
    {
      if (m_buffer.handle != CGPU_INVALID_HANDLE)
      {
        cgpuDestroyBuffer(m_device, m_buffer);
        m_buffer.handle = CGPU_INVALID_HANDLE;
      }

      m_size = 0;
      return true;
    }

    // Create new, larger buffer.
    bool result = false;
    CgpuCommandBuffer commandBuffer = { CGPU_INVALID_HANDLE };
    CgpuBuffer buffer = { CGPU_INVALID_HANDLE };
    CgpuFence fence = { CGPU_INVALID_HANDLE };

    if (!cgpuCreateBuffer(m_device, m_usageFlags, m_memoryProperties, newSize, &buffer))
      goto cleanup;

    // Copy old buffer data if needed.
    if (m_size > 0)
    {
      // TODO: pass command buffer from the outside
      if (!cgpuCreateCommandBuffer(m_device, &commandBuffer))
        goto cleanup;

      if (!cgpuBeginCommandBuffer(commandBuffer))
        goto cleanup;

      if (!cgpuCmdCopyBuffer(commandBuffer, m_buffer, 0, buffer, 0, m_size))
        goto cleanup;

      if (!cgpuEndCommandBuffer(commandBuffer))
        goto cleanup;

      if (!cgpuCreateFence(m_device, &fence))
        goto cleanup;

      if (!cgpuResetFence(m_device, fence))
        goto cleanup;

      if (!cgpuSubmitCommandBuffer(m_device, commandBuffer, fence))
        goto cleanup;

      if (!cgpuWaitForFence(m_device, fence))
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
    // TODO: add to deletion queue instead (with implicitly inserted semaphore)
    if (buffer.handle != CGPU_INVALID_HANDLE)
      cgpuDestroyBuffer(m_device, buffer);
    if (commandBuffer.handle != CGPU_INVALID_HANDLE)
      cgpuDestroyCommandBuffer(m_device, commandBuffer);
    if (fence.handle != CGPU_INVALID_HANDLE)
      cgpuDestroyFence(m_device, fence);

    return result;
  }
}
