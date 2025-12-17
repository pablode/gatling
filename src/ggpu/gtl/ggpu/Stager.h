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

#pragma once

#include <stdint.h>
#include <functional>

#include <gtl/cgpu/Cgpu.h>

namespace gtl
{
  class GgpuStager
  {
  public:
    GgpuStager(CgpuDevice device);
    ~GgpuStager();

    bool allocate();
    void free();

  public:
    bool flush();

    bool stageToBuffer(const uint8_t* src, uint64_t size, CgpuBuffer dst, uint64_t dstOffset = 0);

    bool stageToImage(const uint8_t* src, uint64_t size, CgpuImage dst, uint32_t width, uint32_t height, uint32_t depth = 1, uint32_t bpp = 4);

  private:
    using CopyFunc = std::function<void(uint64_t srcOffset, uint64_t dstOffset, uint64_t size)>;

    bool stage(const uint8_t* src, uint64_t size, CopyFunc copyFunc, uint32_t offsetAlign = 4);

  private:
    CgpuDevice m_device;

    bool m_hasSharedMem;

    uint32_t m_writeableHalf = 0;
    CgpuBuffer m_stagingBuffer;
    CgpuCommandBuffer m_commandBuffers[2];
    CgpuSemaphore m_semaphore;
    uint32_t m_semaphoreCounter = 0;

    bool m_commandsPending = false;
    uint64_t m_stagedBytes = 0;
    uint8_t* m_mappedMem = nullptr;
  };
}
