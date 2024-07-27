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

  bool GgpuResizableBuffer::resize(uint64_t newSize)
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
    CgpuCommandBuffer commandBuffer;
    CgpuBuffer buffer;
    CgpuSemaphore semaphore;
    CgpuSignalSemaphoreInfo signalSemaphoreInfo;
    CgpuWaitSemaphoreInfo waitSemaphoreInfo;

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
      // TODO: pass command buffer from the outside & remove semaphore
      if (!cgpuCreateCommandBuffer(m_device, &commandBuffer))
        goto cleanup;

      if (!cgpuBeginCommandBuffer(commandBuffer))
        goto cleanup;

      if (!cgpuCmdCopyBuffer(commandBuffer, m_buffer, 0, buffer, 0, m_size))
        goto cleanup;

      if (!cgpuEndCommandBuffer(commandBuffer))
        goto cleanup;

      if (!cgpuCreateSemaphore(m_device, &semaphore))
        goto cleanup;

      signalSemaphoreInfo = { .semaphore = semaphore, .value = 1 };
      if (!cgpuSubmitCommandBuffer(m_device, commandBuffer, 1, &signalSemaphoreInfo))
        goto cleanup;

      waitSemaphoreInfo = { .semaphore = semaphore, .value = 1 };
      if (!cgpuWaitSemaphores(m_device, 1, &waitSemaphoreInfo))
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

    if (commandBuffer.handle)
      cgpuDestroyCommandBuffer(m_device, commandBuffer);
    if (semaphore.handle)
      cgpuDestroySemaphore(m_device, semaphore);

    return result;
  }
}
