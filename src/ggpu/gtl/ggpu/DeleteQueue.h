//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#include <gtl/cgpu/Cgpu.h>

#include <functional>
#include <vector>
#include <list>

namespace gtl
{
  class GgpuDeleteQueue
  {
  private:
    constexpr static uint32_t FrameCount = 4;

  public:
    GgpuDeleteQueue(CgpuContext* ctx);

    GgpuDeleteQueue(const GgpuDeleteQueue&) = delete;
    GgpuDeleteQueue& operator=(const GgpuDeleteQueue&) = delete;

    ~GgpuDeleteQueue();

  public:
    void nextFrame();
    void housekeep();

    void destroyAll();

    template<typename T, typename... U>
    void pushBack(T handle, U... moreHandles)
    {
      pushBack(handle);
      pushBack(moreHandles...);
    }

    void pushBack(CgpuBuffer handle);
    void pushBack(CgpuImage handle);
    void pushBack(CgpuPipeline handle);
    void pushBack(CgpuSemaphore handle);
    void pushBack(CgpuCommandBuffer handle);
    void pushBack(CgpuBlas handle);
    void pushBack(CgpuTlas handle);

  private:
    using DestroyFunc = std::function<void()>;

    void enqueueDestroyFunc(DestroyFunc fun);

  private:
    CgpuContext* m_ctx;
    uint32_t m_frameIndex = 0;
    std::vector<DestroyFunc> m_pendingDestructions[FrameCount];
  };
}
