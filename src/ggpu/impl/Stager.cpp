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

const static uint64_t BUFFER_STAGING_OPT_THRESHOLD = 256 * 1024;

namespace gtl
{
  GgpuStager::GgpuStager(CgpuContext* ctx, bool hasSharedMem, uint32_t bufferSize)
    : m_ctx(ctx)
    , m_hasSharedMem(hasSharedMem)
    , m_bufferSize(bufferSize)
  {
    assert(bufferSize > 0);
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
      .size = m_bufferSize,
      .debugName = "Staging"
    };

    bool bufferCreated = cgpuCreateBuffer(m_ctx, createInfo, &m_stagingBuffer);

    if (!bufferCreated)
    {
      createInfo.memoryProperties = CgpuMemoryProperties::HostVisible;

      bufferCreated = cgpuCreateBuffer(m_ctx, createInfo, &m_stagingBuffer);
    }

    if (!bufferCreated)
      goto fail;

    m_commandPools[0] = cgpuCreateCommandPool(m_ctx);
    m_commandPools[1] = cgpuCreateCommandPool(m_ctx);
    cgpuResetCommandPool(m_ctx, m_commandPools[0]);
    cgpuResetCommandPool(m_ctx, m_commandPools[1]);

    if (!cgpuCreateSemaphore(m_ctx, &m_semaphore))
      goto fail;

    m_mappedMem = (uint8_t*) cgpuGetBufferCpuPtr(m_ctx, m_stagingBuffer);

    m_commandBuffers[m_writeableHalf] = cgpuBeginCommandBuffer(m_ctx, m_commandPools[m_writeableHalf]);

    return true;

fail:
    free();
    return false;
  }

  void GgpuStager::free()
  {
    CgpuWaitSemaphoreInfo waitSemaphoreInfo{ .semaphore = m_semaphore, .value = m_semaphoreCounter };
    cgpuWaitSemaphores(m_ctx, 1, &waitSemaphoreInfo);
    cgpuEndCommandBuffer(m_ctx, m_commandBuffers[m_writeableHalf]);
    cgpuDestroySemaphore(m_ctx, m_semaphore);
    cgpuDestroyBuffer(m_ctx, m_stagingBuffer);
    cgpuDestroyCommandPool(m_ctx, m_commandPools[0]);
    cgpuDestroyCommandPool(m_ctx, m_commandPools[1]);
  }

  bool GgpuStager::flush()
  {
    if (m_stagedBytes == 0 && !m_commandsPending)
      return true;

    // Wait until previous submit is finished.
    CgpuWaitSemaphoreInfo waitSemaphoreInfo{ .semaphore = m_semaphore, .value = m_semaphoreCounter };
    if (!cgpuWaitSemaphores(m_ctx, 1, &waitSemaphoreInfo))
      return false;

    m_semaphoreCounter++;

    cgpuEndCommandBuffer(m_ctx, m_commandBuffers[m_writeableHalf]);

    CgpuSignalSemaphoreInfo signalSemaphoreInfo{ .semaphore = m_semaphore, .value = m_semaphoreCounter };
    cgpuSubmitCommandBuffer(m_ctx, m_commandBuffers[m_writeableHalf], 1, &signalSemaphoreInfo);

    m_stagedBytes = 0;
    m_commandsPending = false;
    m_writeableHalf = (m_writeableHalf == 0) ? 1 : 0;

    cgpuResetCommandPool(m_ctx, m_commandPools[m_writeableHalf]);
    m_commandBuffers[m_writeableHalf] = cgpuBeginCommandBuffer(m_ctx, m_commandPools[m_writeableHalf]);

    return true;
  }

  bool GgpuStager::stageToBuffer(const uint8_t* src, uint64_t size, CgpuBuffer dst, uint64_t dstBaseOffset)
  {
    if (size == 0)
    {
      assert(false);
      return true;
    }

    // Use copy command for compression for large data regions
    if (m_hasSharedMem && size < BUFFER_STAGING_OPT_THRESHOLD)
    {
      auto ptr = (uint8_t*) cgpuGetBufferCpuPtr(m_ctx, dst);
      memcpy(&ptr[dstBaseOffset], src, size);
      return true;
    }

    auto copyFunc = [this, dst, dstBaseOffset](uint64_t srcOffset, uint64_t dstOffset, uint64_t size) {
      cgpuCmdCopyBuffer(
        m_ctx,
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
    assert(size > 0 && width > 0 && height > 0 && bpp > 0);

    uint64_t halfSize = m_bufferSize / 2;
    uint32_t rowCount = height;
    uint64_t rowSize = size / rowCount;

    rowSize = rowSize / bpp * bpp; // truncate for Vulkan alignment requirements
    rowCount = size / rowSize;

    if (rowSize > halfSize)
    {
      return false;
    }

    uint32_t rowsStaged = 0;

    while (rowsStaged < rowCount)
    {
      // Account for the alignment padding applied by 'stage', otherwise a row group
      // that barely fits would be split across staging buffer flushes.
      uint64_t padding = (bpp - (m_stagedBytes % bpp)) % bpp;
      uint64_t remainingSpace = halfSize - m_stagedBytes - padding;
      uint32_t maxCopyRowCount = uint32_t(remainingSpace / rowSize); // truncate

      if (maxCopyRowCount == 0)
      {
        if (!flush())
        {
          return false;
        }

        maxCopyRowCount = uint32_t(halfSize / rowSize); // truncate
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
          m_ctx,
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
    uint64_t halfSize = m_bufferSize / 2;
    uint64_t bytesToStage = size;

    while (bytesToStage > 0)
    {
      uint64_t bytesAlreadyCopied = size - bytesToStage;

      uint64_t halfBase = m_writeableHalf * halfSize;
      uint64_t dstOffset = (halfBase + m_stagedBytes + offsetAlign - 1) / offsetAlign * offsetAlign;
      uint64_t padding = dstOffset - halfBase - m_stagedBytes;

      if ((m_stagedBytes + padding) >= halfSize)
      {
        if (!flush())
        {
          return false;
        }
        continue;
      }

      m_stagedBytes += padding;

      uint64_t memcpyByteCount = std::min(bytesToStage, halfSize - m_stagedBytes);

      memcpy(&m_mappedMem[dstOffset], &src[bytesAlreadyCopied], memcpyByteCount);

      copyFunc(dstOffset, bytesAlreadyCopied, memcpyByteCount);

      m_commandsPending = true;
      m_stagedBytes += memcpyByteCount;
      bytesToStage -= memcpyByteCount;

      if (m_stagedBytes == halfSize && !flush())
      {
        return false;
      }
    }

    return true;
  }
}

#ifdef GGPU_TEST_EXECUTABLE
#include <doctest/doctest.h>
#include <memory>
#include <vector>

namespace
{
  using namespace gtl;

  const uint32_t DEFAULT_BUFFER_SIZE = 1024 * 1024;
  const uint32_t SMALL_BUFFER_SIZE = 128;
  const uint8_t INITIAL_READBACK_VALUE = 0xCC;

  struct StagerTestFixture
  {
    CgpuContext* ctx;
    std::unique_ptr<GgpuStager> stager;
    uint8_t* readbackData;
    CgpuBuffer readbackBuffer;

    void initialize(uint32_t stagerBufferSize = DEFAULT_BUFFER_SIZE, bool hasSharedMem = false)
    {
      ctx = cgpuCreateContext("ggpu_test");
      REQUIRE(ctx);

      stager = std::make_unique<GgpuStager>(ctx, hasSharedMem, stagerBufferSize);
      REQUIRE(stager->allocate());

      CgpuBufferCreateInfo readbackCreateInfo = {
        .usage = CgpuBufferUsage::TransferDst,
        .memoryProperties = CgpuMemoryProperties::HostVisible,
        .size = DEFAULT_BUFFER_SIZE,
        .debugName = "Stager test readback"
      };
      REQUIRE(cgpuCreateBuffer(ctx, readbackCreateInfo, &readbackBuffer));

      readbackData = (uint8_t*) cgpuGetBufferCpuPtr(ctx, readbackBuffer);
      REQUIRE(readbackData);
    }

    ~StagerTestFixture()
    {
      stager->flush();
      stager->free();
      cgpuDestroyBuffer(ctx, readbackBuffer);
      cgpuDestroyContext(ctx);
    }

    void submitAndWait(CgpuCommandPool commandPool, CgpuCommandBuffer commandBuffer)
    {
      CgpuSemaphore semaphore;
      REQUIRE(cgpuCreateSemaphore(ctx, &semaphore));

      CgpuSignalSemaphoreInfo signalInfo = {
        .semaphore = semaphore,
        .value = 1
      };
      cgpuSubmitCommandBuffer(ctx, commandBuffer, 1, &signalInfo);

      CgpuWaitSemaphoreInfo waitInfo = {
        .semaphore = semaphore,
        .value = 1
      };
      REQUIRE(cgpuWaitSemaphores(ctx, 1, &waitInfo));

      cgpuDestroySemaphore(ctx, semaphore);
      cgpuDestroyCommandPool(ctx, commandPool);
    }

    void fullFlush()
    {
      REQUIRE(stager->flush());

      CgpuCommandPool commandPool = cgpuCreateCommandPool(ctx);
      CgpuCommandBuffer commandBuffer = cgpuBeginCommandBuffer(ctx, commandPool);
      cgpuEndCommandBuffer(ctx, commandBuffer);

      submitAndWait(commandPool, commandBuffer);
    }

    void copyImageToBuffer(CgpuImage image, uint32_t width, uint32_t height)
    {
      CgpuCommandPool commandPool = cgpuCreateCommandPool(ctx);
      CgpuCommandBuffer commandBuffer = cgpuBeginCommandBuffer(ctx, commandPool);

      CgpuBufferImageCopyDesc desc = {
        .texelExtentX = width,
        .texelExtentY = height,
        .texelExtentZ = 1
      };
      cgpuCmdCopyImageToBuffer(ctx, commandBuffer, image, readbackBuffer, &desc);

      cgpuEndCommandBuffer(ctx, commandBuffer);

      submitAndWait(commandPool, commandBuffer);
    }
  };

  struct SimpleStagerTestFixture : StagerTestFixture
  {
    SimpleStagerTestFixture() { initialize(); }
  };

  void _StageToBufferAlignsCopies(StagerTestFixture& fixture)
  {
    const uint64_t bufferSize = 32;
    const uint64_t copyOffset = 16;

    memset(fixture.readbackData, INITIAL_READBACK_VALUE, bufferSize);

    const uint8_t first[] = { 1, 2, 3, 4, 5 };
    const uint8_t second[] = { 6, 7, 8, 9 };

    CHECK(fixture.stager->stageToBuffer(first, sizeof(first), fixture.readbackBuffer, 0));
    CHECK(fixture.stager->stageToBuffer(second, sizeof(second), fixture.readbackBuffer, copyOffset));
    fixture.fullFlush();

    CHECK(memcmp(fixture.readbackData, first, sizeof(first)) == 0);
    CHECK(memcmp(fixture.readbackData + copyOffset, second, sizeof(second)) == 0);
  }

  void _StageToBufferSpansHalves(StagerTestFixture& fixture)
  {
    const uint64_t dataSize = (DEFAULT_BUFFER_SIZE / 2) + 16;

    std::vector<uint8_t> data(dataSize);
    for (size_t i = 0; i < data.size(); i++)
    {
      data[i] = uint8_t(i);
    }

    CHECK(fixture.stager->stageToBuffer(data.data(), data.size(), fixture.readbackBuffer));
    fixture.fullFlush();

    CHECK(memcmp(fixture.readbackData, data.data(), data.size()) == 0);
  }

  void _StageToBufferMovesMisalignedCopyToNextHalf(StagerTestFixture& fixture)
  {
    const uint64_t fillSize = (DEFAULT_BUFFER_SIZE / 2) - 2;

    std::vector<uint8_t> fill(fillSize, 170);
    CHECK(fixture.stager->stageToBuffer(fill.data(), fill.size(), fixture.readbackBuffer));

    const uint8_t data[] = { 1, 2, 3, 4 };
    CHECK(fixture.stager->stageToBuffer(data, sizeof(data), fixture.readbackBuffer));
    fixture.fullFlush();

    CHECK(memcmp(fixture.readbackData, data, sizeof(data)) == 0);
  }

  void _StageToImage(StagerTestFixture& fixture, uint32_t width, uint32_t height)
  {
    const uint32_t bpp = 4;

    CgpuImage image = {};
    CgpuImageCreateInfo imageInfo = {
      .width = width,
      .height = height,
      .format = CgpuImageFormat::R8G8B8A8Unorm,
      .usage = CgpuImageUsage::TransferSrc | CgpuImageUsage::TransferDst
    };
    REQUIRE(cgpuCreateImage(fixture.ctx, imageInfo, &image));

    std::vector<uint8_t> pixels(width * height * bpp);
    for (size_t i = 0; i < pixels.size(); i++)
    {
      pixels[i] = uint8_t(i);
    }

    CHECK(fixture.stager->stageToImage(pixels.data(), pixels.size(), image, width, height, 1, bpp));
    fixture.fullFlush();
    fixture.copyImageToBuffer(image, width, height);

    CHECK(memcmp(fixture.readbackData, pixels.data(), pixels.size()) == 0);

    cgpuDestroyImage(fixture.ctx, image);
  }
}

TEST_CASE_FIXTURE(SimpleStagerTestFixture, "Stager.StageToBufferAlignsCopies")
{
  _StageToBufferAlignsCopies(*this);
}

TEST_CASE_FIXTURE(StagerTestFixture, "Stager.StageToBufferAlignsCopiesSharedMem")
{
  bool hasSharedMem = true;
  initialize(DEFAULT_BUFFER_SIZE, hasSharedMem);

  _StageToBufferAlignsCopies(*this);
}

TEST_CASE_FIXTURE(SimpleStagerTestFixture, "Stager.StageToBufferSpansHalves")
{
  _StageToBufferSpansHalves(*this);
}

TEST_CASE_FIXTURE(StagerTestFixture, "Stager.StageToBufferSpansHalvesSharedMem")
{
  bool hasSharedMem = true;
  initialize(DEFAULT_BUFFER_SIZE, hasSharedMem);

  _StageToBufferSpansHalves(*this);
}

TEST_CASE_FIXTURE(SimpleStagerTestFixture, "Stager.StageToBufferMovesMisalignedCopyToNextHalf")
{
  _StageToBufferMovesMisalignedCopyToNextHalf(*this);
}

TEST_CASE_FIXTURE(SimpleStagerTestFixture, "Stager.StageToImage")
{
  uint32_t width = 2;
  uint32_t height = 4;
  _StageToImage(*this, width, height);
}

TEST_CASE_FIXTURE(StagerTestFixture, "Stager.StageToImageChunks")
{
  initialize(SMALL_BUFFER_SIZE);

  uint32_t width = 13; // primes on purpose
  uint32_t height = 9;
  _StageToImage(*this, width, height);
}
#endif
