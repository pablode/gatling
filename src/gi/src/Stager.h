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

#include <cgpu.h>

#include <stdint.h>
#include <functional>

namespace gtl
{
  class GiStager
  {
  public:
    GiStager(CgpuDevice device);
    ~GiStager();

    bool allocate();
    void free();

  public:
    bool flush();

    bool stageToBuffer(const uint8_t* src, uint64_t size, CgpuBuffer dst, uint64_t dstOffset);

    bool stageToImage(const uint8_t* src, uint64_t size, CgpuImage dst, uint32_t width, uint32_t height, uint32_t depth);

  private:
    using CopyFunc = std::function<bool(uint64_t srcOffset, uint64_t dstOffset, uint64_t size)>;

    bool stage(const uint8_t* src, uint64_t size, CopyFunc copyFunc);

  private:
    CgpuDevice m_device;

    CgpuBuffer m_stagingBuffer = { CGPU_INVALID_HANDLE };
    CgpuCommandBuffer m_commandBuffer = { CGPU_INVALID_HANDLE };
    CgpuFence m_fence = { CGPU_INVALID_HANDLE };

    uint64_t m_stagedBytes = 0;
    uint8_t* m_mappedMem = nullptr;
  };
}
