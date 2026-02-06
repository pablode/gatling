//
// Copyright (C) 2025 Pablo Delgado Krämer
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

#include <gtl/gb/Class.h>
#include <gtl/cgpu/Cgpu.h>

#include <memory>

namespace gtl
{
  class GgpuDeleteQueue;

  template<typename T>
  struct GgpuTempAllocation
  {
    T* cpuPtr;
    uint32_t bufferOffset;
  };

  // Inspired by this blog post: https://www.sebastianaaltonen.com/blog/no-graphics-api
  class GgpuBumpAllocator
  {
  public:
    static std::shared_ptr<GgpuBumpAllocator> make(CgpuContext* ctx, GgpuDeleteQueue& deleteQueue, uint32_t size);

  public:
    GB_DECLARE_NONCOPY(GgpuBumpAllocator);

    GgpuBumpAllocator(CgpuContext* ctx,
                      GgpuDeleteQueue& deleteQueue,
                      CgpuBuffer buffer, uint32_t size);

    ~GgpuBumpAllocator();

    CgpuBuffer getBuffer() const;

    template<typename T>
    GgpuTempAllocation<T> alloc(uint32_t count = 1)
    {
      GgpuTempAllocation<uint8_t> tmp = alloc(sizeof(T) * count);
      return GgpuTempAllocation<T> { (T*) tmp.cpuPtr, tmp.bufferOffset };
    }

  private:
    GgpuTempAllocation<uint8_t> alloc(uint32_t size);

  private:
    GgpuDeleteQueue& m_deleteQueue;
    CgpuBuffer m_buffer;
    uint8_t* m_cpuPtr;
    uint32_t m_offset;
    uint32_t m_size;
    uint32_t m_align;
  };
}
