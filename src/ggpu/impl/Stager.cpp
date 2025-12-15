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
    const CgpuDeviceFeatures& features = cgpuGetDeviceFeatures(device);

    m_useRebar = features.rebar;
  }

  GgpuStager::~GgpuStager()
  {
    // Ensure data has been flushed.
    assert(m_stagedBytes == 0);
  }

  bool GgpuStager::allocate()
  {
    CgpuBufferCreateInfo createInfo = {
      .usage = CgpuBufferUsage::TransferSrc,
      .memoryProperties = CgpuMemoryProperties::DeviceLocal | CgpuMemoryProperties::HostVisible,
      .size = BUFFER_SIZE,
      .debugName = "Staging"
    };

    bool bufferCreated = cgpuCreateBuffer(m_device, createInfo, &m_stagingBuffer);

    if (!bufferCreated)
    {
      createInfo.memoryProperties = CgpuMemoryProperties::HostVisible | CgpuMemoryProperties::HostCached;

      bufferCreated = cgpuCreateBuffer(m_device, createInfo, &m_stagingBuffer);
    }

    if (!bufferCreated)
      goto fail;

    if (!cgpuCreateCommandBuffer(m_device, &m_commandBuffers[0]) ||
        !cgpuCreateCommandBuffer(m_device, &m_commandBuffers[1]))
      goto fail;

    if (!cgpuCreateSemaphore(m_device, &m_semaphore))
      goto fail;

    cgpuMapBuffer(m_device, m_stagingBuffer, (void**) &m_mappedMem);

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

    cgpuEndCommandBuffer(m_commandBuffers[m_writeableHalf]);

    uint32_t halfOffset = m_writeableHalf * BUFFER_HALF_SIZE;
    cgpuFlushMappedMemory(m_device, m_stagingBuffer, halfOffset, halfOffset + m_stagedBytes);

    CgpuSignalSemaphoreInfo signalSemaphoreInfo{ .semaphore = m_semaphore, .value = m_semaphoreCounter };
    cgpuSubmitCommandBuffer(m_device, m_commandBuffers[m_writeableHalf], 1, &signalSemaphoreInfo);

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

    if (m_useRebar)
    {
      uint8_t* mappedMem;
      cgpuMapBuffer(m_device, dst, (void**) &mappedMem);

      memcpy(&mappedMem[dstBaseOffset], src, size);

      cgpuUnmapBuffer(m_device, dst);
      return true;
    }

    if (size <= CGPU_MAX_BUFFER_UPDATE_SIZE)
    {
      m_commandsPending = true;

      cgpuCmdUpdateBuffer(m_commandBuffers[m_writeableHalf], src, size, dst, dstBaseOffset);

      return true;
    }

    auto copyFunc = [this, dst, dstBaseOffset](uint64_t srcOffset, uint64_t dstOffset, uint64_t size) {
      cgpuCmdCopyBuffer(
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

  bool GgpuStager::stageToImage(const uint8_t* src, uint64_t size, CgpuImage dst, uint32_t width, uint32_t height, uint32_t depth, uint32_t bpp)
  {
    uint32_t rowCount = height;
    uint64_t rowSize = size / rowCount;

    rowSize = rowSize / bpp * bpp; // truncate for Vulkan alignment requirements
    rowCount = size / rowSize;

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
        desc.bufferOffset = srcOffset; // Vulkan requirement: must be multiple of texel format
        desc.texelOffsetX = 0;
        desc.texelExtentX = width;
        desc.texelOffsetY = rowsStaged;
        desc.texelExtentY = copyRowCount;
        desc.texelOffsetZ = 0;
        desc.texelExtentZ = depth;

        cgpuCmdCopyBufferToImage(
          m_commandBuffers[m_writeableHalf],
          m_stagingBuffer,
          dst,
          &desc
        );
      };

      uint64_t srcOffset = rowsStaged * rowSize;
      uint64_t stageSize = copyRowCount * rowSize;
      if (!stage(&src[srcOffset], stageSize, copyFunc, bpp))
      {
        return false;
      }

      rowsStaged += copyRowCount;
    }

    return true;
  }

  bool GgpuStager::stage(const uint8_t* src, uint64_t size, CopyFunc copyFunc, uint32_t offsetAlign)
  {
    uint64_t bytesToStage = size;

    while (bytesToStage > 0)
    {
      uint64_t bytesAlreadyCopied = size - bytesToStage;
      uint64_t requiredSpace = (bytesToStage + offsetAlign - 1) / offsetAlign * offsetAlign;

      uint64_t availableSpace = BUFFER_HALF_SIZE - m_stagedBytes;
      uint64_t memcpyByteCount = std::min(requiredSpace, availableSpace);
      bytesToStage -= memcpyByteCount;

      uint64_t dstOffset = m_writeableHalf * BUFFER_HALF_SIZE + m_stagedBytes;
      dstOffset = (dstOffset + offsetAlign - 1) / offsetAlign * offsetAlign;

      memcpy(&m_mappedMem[dstOffset], &src[bytesAlreadyCopied], memcpyByteCount);

      copyFunc(dstOffset, bytesAlreadyCopied, memcpyByteCount);

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
