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

#include "linearDataStoreGpu.h"

namespace
{
  // See: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  uint32_t _NextPowerOfTwo(uint32_t v)
  {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
  }
}

namespace gtl
{
  GiLinearDataStoreGpu::GiLinearDataStoreGpu(CgpuDevice device, uint64_t elementSize, uint32_t initialCapacity)
    : m_device(device)
    , m_elementSize(elementSize)
  {
    resizeBuffer(elementSize * initialCapacity);
  }

  GiLinearDataStoreGpu::~GiLinearDataStoreGpu()
  {
    if (m_mappedMem)
      cgpuUnmapBuffer(m_device, m_buffer);

    // TODO: add to deletion queue instead
    if (m_buffer.handle != CGPU_INVALID_HANDLE)
      cgpuDestroyBuffer(m_device, m_buffer);
  }

  uint64_t GiLinearDataStoreGpu::allocate()
  {
    return m_handleStore.allocateHandle();
  }

  void GiLinearDataStoreGpu::free(uint64_t handle)
  {
    m_handleStore.freeHandle(handle);
  }

  bool GiLinearDataStoreGpu::get(uint64_t handle, void** element)
  {
    if (!m_handleStore.isHandleValid(handle))
    {
      assert(false);
      return false;
    }

    uint32_t index = uint32_t(handle);
    uint64_t byteOffset = index * m_elementSize;

    if (byteOffset >= m_bufferSize)
    {
      uint32_t newSize = _NextPowerOfTwo(byteOffset);

      if (!resizeBuffer(newSize))
      {
        assert(false);
        return false;
      }
    }

    *element = (void*) &m_mappedMem[byteOffset];
    return true;
  }

  CgpuBuffer GiLinearDataStoreGpu::buffer() const
  {
    return m_buffer;
  }

  uint64_t GiLinearDataStoreGpu::bufferSize() const
  {
    return m_bufferSize;
  }

  bool GiLinearDataStoreGpu::resizeBuffer(uint64_t newSize)
  {
    // Unmap buffer before resize. New buffer is mapped later.
    if (m_mappedMem)
    {
      cgpuUnmapBuffer(m_device, m_buffer);
    }

    // Create new, larger buffer.
    bool result = false;
    CgpuCommandBuffer commandBuffer = { CGPU_INVALID_HANDLE };
    CgpuBuffer buffer = { CGPU_INVALID_HANDLE };
    CgpuFence fence = { CGPU_INVALID_HANDLE };

    CgpuBufferUsageFlags bufferUsageFlags = CGPU_BUFFER_USAGE_FLAG_STORAGE_BUFFER |
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC | CGPU_BUFFER_USAGE_FLAG_TRANSFER_DST;

    CgpuMemoryPropertyFlags bufferMemoryProperties =
      CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL | CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE;

    if (!cgpuCreateBuffer(m_device, bufferUsageFlags, bufferMemoryProperties, newSize, &buffer))
      goto cleanup;

    // Copy old buffer data if needed.
    if (m_bufferSize > 0)
    {
      if (!cgpuCreateCommandBuffer(m_device, &commandBuffer))
        goto cleanup;

      if (!cgpuBeginCommandBuffer(commandBuffer))
        goto cleanup;

      if (!cgpuCmdCopyBuffer(commandBuffer, m_buffer, 0, buffer, 0, m_bufferSize))
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

    // Persistently map buffer.
    if (!cgpuMapBuffer(m_device, buffer, (void**) &m_mappedMem))
      goto cleanup;

    // Swap buffers, so that we always destroy the unused one.
    {
      CgpuBuffer temp = m_buffer;
      m_buffer = buffer;
      buffer = temp;
    }

    m_bufferSize = newSize;

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
