//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
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

#include "stager.h"

#include <assert.h>
#include <algorithm>

const static uint64_t BUFFER_SIZE = 64 * 1024 * 1024;

namespace gi
{
  Stager::Stager(cgpu_device device)
    : m_device(device)
  {
  }

  Stager::~Stager()
  {
    assert(m_stagedBytes == 0);
  }

  bool Stager::allocate()
  {
    m_stagingBuffer = { CGPU_INVALID_HANDLE };
    m_commandBuffer = { CGPU_INVALID_HANDLE };
    m_fence = { CGPU_INVALID_HANDLE };

    // Try to use ReBAR if available.
    CgpuResult c_result = cgpu_create_buffer(m_device,
      CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
      CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL | CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE,
      BUFFER_SIZE,
      &m_stagingBuffer
    );

    if (c_result != CGPU_OK)
    {
      c_result = cgpu_create_buffer(m_device,
        CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
        CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED,
        BUFFER_SIZE,
        &m_stagingBuffer
      );
      if (c_result != CGPU_OK) goto fail;
    }

    c_result = cgpu_create_command_buffer(m_device, &m_commandBuffer);
    if (c_result != CGPU_OK) goto fail;

    c_result = cgpu_create_fence(m_device, &m_fence);
    if (c_result != CGPU_OK) goto fail;

    c_result = cgpu_map_buffer(m_device, m_stagingBuffer, (void**) &m_mappedMem);
    if (c_result != CGPU_OK) goto fail;

    c_result = cgpu_begin_command_buffer(m_commandBuffer);
    if (c_result != CGPU_OK) goto fail;

    return true;

fail:
    free();
    return false;
  }

  void Stager::free()
  {
    assert(m_stagedBytes == 0);
    cgpu_end_command_buffer(m_commandBuffer);
    if (m_mappedMem)
    {
      cgpu_unmap_buffer(m_device, m_stagingBuffer);
    }
    cgpu_destroy_fence(m_device, m_fence);
    cgpu_destroy_command_buffer(m_device, m_commandBuffer);
    cgpu_destroy_buffer(m_device, m_stagingBuffer);
  }

  bool Stager::flush()
  {
    if (m_stagedBytes == 0)
    {
      return true;
    }

    CgpuResult c_result;

    c_result = cgpu_flush_mapped_memory(m_device, m_stagingBuffer, 0, m_stagedBytes);
    if (c_result != CGPU_OK) return false;

    c_result = cgpu_reset_fence(m_device, m_fence);
    if (c_result != CGPU_OK) return false;

    c_result = cgpu_end_command_buffer(m_commandBuffer);
    if (c_result != CGPU_OK) return false;

    c_result = cgpu_submit_command_buffer(m_device, m_commandBuffer, m_fence);
    if (c_result != CGPU_OK) return false;

    c_result = cgpu_wait_for_fence(m_device, m_fence);
    if (c_result != CGPU_OK) return false;

    c_result = cgpu_begin_command_buffer(m_commandBuffer);
    if (c_result != CGPU_OK) return false;

    m_stagedBytes = 0;
    return true;
  }

  bool Stager::stageToBuffer(const uint8_t* src, uint64_t size, cgpu_buffer dst, uint64_t dstBaseOffset)
  {
    auto copyFunc = [this, dst, dstBaseOffset](uint64_t srcOffset, uint64_t dstOffset, uint64_t size) {
      return cgpu_cmd_copy_buffer(
        m_commandBuffer,
        m_stagingBuffer,
        srcOffset,
        dst,
        dstBaseOffset + dstOffset,
        size
      );
    };

    return stage(src, size, copyFunc);
  }

  bool Stager::stageToImage(const uint8_t* src, uint64_t size, cgpu_image dst)
  {
    if (size > BUFFER_SIZE)
    {
      return false;
    }

    uint64_t availableSpace = BUFFER_SIZE - m_stagedBytes;

    if (availableSpace < size)
    {
      // We don't partially copy to an image, so we need to make sure everything fits into the staging buffer.
      if (!flush())
      {
        return false;
      }
    }

    auto copyFunc = [this, dst](uint64_t srcOffset, uint64_t dstOffset, uint64_t size) {
      return cgpu_cmd_copy_buffer_to_image(
        m_commandBuffer,
        m_stagingBuffer,
        srcOffset,
        dst
      );
    };

    return stage(src, size, copyFunc);
  }

  bool Stager::stage(const uint8_t* src, uint64_t size, CopyFunc copyFunc)
  {
    uint64_t bytesToStage = size;

    while (bytesToStage > 0)
    {
      uint64_t bytesAlreadyCopied = size - bytesToStage;

      uint64_t availableSpace = BUFFER_SIZE - m_stagedBytes;
      uint64_t memcpyByteCount = std::min(bytesToStage, availableSpace);
      bytesToStage -= memcpyByteCount;

      memcpy(&m_mappedMem[m_stagedBytes], &src[bytesAlreadyCopied], memcpyByteCount);

      CgpuResult c_result = copyFunc(m_stagedBytes, bytesAlreadyCopied, memcpyByteCount);
      if (c_result != CGPU_OK) return false;

      m_stagedBytes += memcpyByteCount;

      if (m_stagedBytes != BUFFER_SIZE)
      {
        continue;
      }

      if (flush())
      {
        continue;
      }

      return false;
    }

    return true;
  }
}
