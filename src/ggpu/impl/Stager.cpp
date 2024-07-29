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

#include "Stager.h"

#include <assert.h>
#include <string.h>
#include <algorithm>

const static uint64_t BUFFER_SIZE = 64 * 1024 * 1024;
const static uint64_t BUFFER_HALF_SIZE = BUFFER_SIZE / 2;

namespace gtl
{
  GgpuStager::GgpuStager(CgpuDevice device)
    : m_device(device)
  {
  }

  GgpuStager::~GgpuStager()
  {
    // Ensure data has been flushed.
    assert(m_stagedBytes == 0);
  }

  bool GgpuStager::allocate()
  {
    CgpuBufferCreateInfo createInfo = {
      .usage = CGPU_BUFFER_USAGE_FLAG_TRANSFER_SRC,
      .memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_DEVICE_LOCAL | CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE,
      .size = BUFFER_SIZE,
      .debugName = "Staging"
    };

    bool bufferCreated = cgpuCreateBuffer(m_device, createInfo, &m_stagingBuffer);

    if (!bufferCreated)
    {
      createInfo.memoryProperties = CGPU_MEMORY_PROPERTY_FLAG_HOST_VISIBLE | CGPU_MEMORY_PROPERTY_FLAG_HOST_CACHED;

      bufferCreated = cgpuCreateBuffer(m_device, createInfo, &m_stagingBuffer);
    }

    if (!bufferCreated)
      goto fail;

    if (!cgpuCreateCommandBuffer(m_device, &m_commandBuffers[0]) ||
        !cgpuCreateCommandBuffer(m_device, &m_commandBuffers[1]))
      goto fail;

    if (!cgpuCreateSemaphore(m_device, &m_semaphore))
      goto fail;

    if (!cgpuMapBuffer(m_device, m_stagingBuffer, (void**) &m_mappedMem))
      goto fail;

    if (!cgpuBeginCommandBuffer(m_commandBuffers[m_writeableHalf]))
      goto fail;

    return true;

fail:
    free();
    return false;
  }

  void GgpuStager::free()
  {
    CgpuWaitSemaphoreInfo waitSemaphoreInfo{ .semaphore = m_semaphore, .value = m_semaphoreCounter };
    cgpuWaitSemaphores(m_device, 1, &waitSemaphoreInfo);
    if (m_mappedMem)
    {
      cgpuUnmapBuffer(m_device, m_stagingBuffer);
    }
    cgpuEndCommandBuffer(m_commandBuffers[m_writeableHalf]);
    cgpuDestroySemaphore(m_device, m_semaphore);
    cgpuDestroyCommandBuffer(m_device, m_commandBuffers[0]);
    cgpuDestroyCommandBuffer(m_device, m_commandBuffers[1]);
    cgpuDestroyBuffer(m_device, m_stagingBuffer);
  }

  bool GgpuStager::flush()
  {
    if (m_stagedBytes == 0 && !m_commandsPending)
      return true;

    // Wait until previous submit is finished.
    CgpuWaitSemaphoreInfo waitSemaphoreInfo{ .semaphore = m_semaphore, .value = m_semaphoreCounter };
    if (!cgpuWaitSemaphores(m_device, 1, &waitSemaphoreInfo))
      return false;

    m_semaphoreCounter++;

    if (!cgpuEndCommandBuffer(m_commandBuffers[m_writeableHalf]))
      return false;

    uint32_t halfOffset = m_writeableHalf * BUFFER_HALF_SIZE;
    if (!cgpuFlushMappedMemory(m_device, m_stagingBuffer, halfOffset, halfOffset + m_stagedBytes))
      return false;

    CgpuSignalSemaphoreInfo signalSemaphoreInfo{ .semaphore = m_semaphore, .value = m_semaphoreCounter };
    if (!cgpuSubmitCommandBuffer(m_device, m_commandBuffers[m_writeableHalf], 1, &signalSemaphoreInfo))
      return false;

    m_stagedBytes = 0;
    m_commandsPending = false;
    m_writeableHalf = (m_writeableHalf == 0) ? 1 : 0;

    if (!cgpuBeginCommandBuffer(m_commandBuffers[m_writeableHalf]))
      return false;

    return true;
  }

  bool GgpuStager::stageToBuffer(const uint8_t* src, uint64_t size, CgpuBuffer dst, uint64_t dstBaseOffset)
  {
    if (size == 0)
    {
      assert(false);
      return true;
    }

    if (size <= 65535)
    {
      m_commandsPending = true;

      return cgpuCmdUpdateBuffer(m_commandBuffers[m_writeableHalf], src, size, dst, dstBaseOffset);
    }

    auto copyFunc = [this, dst, dstBaseOffset](uint64_t srcOffset, uint64_t dstOffset, uint64_t size) {
      return cgpuCmdCopyBuffer(
        m_commandBuffers[m_writeableHalf],
        m_stagingBuffer,
        srcOffset,
        dst,
        dstBaseOffset + dstOffset,
        size
      );
    };

    return stage(src, size, copyFunc);
  }

  bool GgpuStager::stageToImage(const uint8_t* src, uint64_t size, CgpuImage dst, uint32_t width, uint32_t height, uint32_t depth)
  {
    uint32_t rowCount = height;
    uint64_t rowSize = size / rowCount;

    if (rowSize > BUFFER_HALF_SIZE)
    {
      return false;
    }

    uint32_t rowsStaged = 0;

    while (rowsStaged < rowCount)
    {
      uint64_t remainingSpace = BUFFER_HALF_SIZE - m_stagedBytes;
      uint32_t maxCopyRowCount = uint32_t(remainingSpace / rowSize); // truncate

      if (maxCopyRowCount == 0)
      {
        if (!flush())
        {
          return false;
        }

        maxCopyRowCount = uint32_t(BUFFER_HALF_SIZE / rowSize); // truncate
      }

      uint32_t remainingRowCount = rowCount - rowsStaged;
      uint32_t copyRowCount = std::min(remainingRowCount, maxCopyRowCount);

      auto copyFunc = [this, dst, rowsStaged, width, depth, copyRowCount](uint64_t srcOffset, [[maybe_unused]] uint64_t dstOffset, [[maybe_unused]] uint64_t size) {
        CgpuBufferImageCopyDesc desc;
        desc.bufferOffset = srcOffset;
        desc.texelOffsetX = 0;
        desc.texelExtentX = width;
        desc.texelOffsetY = rowsStaged;
        desc.texelExtentY = copyRowCount;
        desc.texelOffsetZ = 0;
        desc.texelExtentZ = depth;

        return cgpuCmdCopyBufferToImage(
          m_commandBuffers[m_writeableHalf],
          m_stagingBuffer,
          dst,
          &desc
        );
      };

      uint64_t srcOffset = rowsStaged * rowSize;
      uint64_t stageSize = copyRowCount * rowSize;
      if (!stage(&src[srcOffset], stageSize, copyFunc))
      {
        return false;
      }

      rowsStaged += copyRowCount;
    }

    return true;
  }

  bool GgpuStager::stage(const uint8_t* src, uint64_t size, CopyFunc copyFunc)
  {
    uint64_t bytesToStage = size;

    while (bytesToStage > 0)
    {
      uint64_t bytesAlreadyCopied = size - bytesToStage;

      uint64_t availableSpace = BUFFER_HALF_SIZE - m_stagedBytes;
      uint64_t memcpyByteCount = std::min(bytesToStage, availableSpace);
      bytesToStage -= memcpyByteCount;

      uint64_t dstOffset = m_writeableHalf * BUFFER_HALF_SIZE + m_stagedBytes;
      memcpy(&m_mappedMem[dstOffset], &src[bytesAlreadyCopied], memcpyByteCount);

      if (!copyFunc(dstOffset, bytesAlreadyCopied, memcpyByteCount))
      {
        return false;
      }

      m_commandsPending = true;
      m_stagedBytes += memcpyByteCount;

      if (m_stagedBytes != BUFFER_HALF_SIZE)
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
