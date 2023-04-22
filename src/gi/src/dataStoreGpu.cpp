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

#include "dataStoreGpu.h"

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
  GiDataStoreGpu::GiDataStoreGpu(CgpuDevice device, uint64_t objectSize, uint32_t initialCapacity)
    : m_device(device)
    , m_objectSize(objectSize)
  {
    resizeBuffer(objectSize * initialCapacity);
  }

  GiDataStoreGpu::~GiDataStoreGpu()
  {
    if (m_mappedMem)
      cgpu_unmap_buffer(m_device, m_buffer);

    // TODO: add to deletion queue instead
    if (m_buffer.handle != CGPU_INVALID_HANDLE)
      cgpu_destroy_buffer(m_device, m_buffer);
  }

  uint64_t GiDataStoreGpu::allocate()
  {
    return m_handleStore.allocateHandle();
  }

  void GiDataStoreGpu::free(uint64_t handle)
  {
    m_handleStore.freeHandle(handle);
  }

  bool GiDataStoreGpu::get(uint64_t handle, void** object)
  {
    if (!m_handleStore.isHandleValid(handle))
    {
      assert(false);
      return false;
    }

    uint32_t index = uint32_t(handle);
    uint64_t byteOffset = index * m_objectSize;

    if (byteOffset >= m_bufferSize)
    {
      uint32_t newSize = _NextPowerOfTwo(byteOffset);

      if (!resizeBuffer(newSize))
      {
        assert(false);
        return false;
      }
    }

    *object = (void*) &m_mappedMem[byteOffset];
    return true;
  }

  CgpuBuffer GiDataStoreGpu::buffer() const
  {
    return m_buffer;
  }

  uint64_t GiDataStoreGpu::bufferSize() const
  {
    return m_bufferSize;
  }

  bool GiDataStoreGpu::resizeBuffer(uint64_t newSize)
  {
    // Unmap buffer before resize. New buffer is mapped later.
    if (m_mappedMem)
    {
      cgpu_unmap_buffer(m_device, m_buffer);
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

    if (!cgpu_create_buffer(m_device, bufferUsageFlags, bufferMemoryProperties, newSize, &buffer))
      goto cleanup;

    // Copy old buffer data if needed.
    if (m_bufferSize > 0)
    {
      if (!cgpu_create_command_buffer(m_device, &commandBuffer))
        goto cleanup;

      if (!cgpu_begin_command_buffer(commandBuffer))
        goto cleanup;

      if (!cgpu_cmd_copy_buffer(commandBuffer, m_buffer, 0, buffer, 0, m_bufferSize))
        goto cleanup;

      if (!cgpu_end_command_buffer(commandBuffer))
        goto cleanup;

      if (!cgpu_create_fence(m_device, &fence))
        goto cleanup;

      if (!cgpu_reset_fence(m_device, fence))
        goto cleanup;

      if (!cgpu_submit_command_buffer(m_device, commandBuffer, fence))
        goto cleanup;

      if (!cgpu_wait_for_fence(m_device, fence))
        goto cleanup;
    }

    // Persistently map buffer.
    if (!cgpu_map_buffer(m_device, buffer, (void**) &m_mappedMem))
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
      cgpu_destroy_buffer(m_device, buffer);
    if (commandBuffer.handle != CGPU_INVALID_HANDLE)
      cgpu_destroy_command_buffer(m_device, commandBuffer);
    if (fence.handle != CGPU_INVALID_HANDLE)
      cgpu_destroy_fence(m_device, fence);

    return result;
  }
}
