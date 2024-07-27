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
  class GgpuDelayedResourceDestroyer
  {
  private:
    constexpr static uint32_t FrameCount = 4;

  public:
    GgpuDelayedResourceDestroyer(CgpuDevice device);

    ~GgpuDelayedResourceDestroyer();

  public:
    void nextFrame();

    void destroyAll();

    template<typename T, typename... U>
    void enqueueDestruction(T handle, U... moreHandles)
    {
      enqueueDestruction(handle);
      enqueueDestruction(moreHandles...);
    }

    void enqueueDestruction(CgpuBuffer handle);
    void enqueueDestruction(CgpuImage handle);
    void enqueueDestruction(CgpuPipeline handle);
    void enqueueDestruction(CgpuSemaphore handle);
    void enqueueDestruction(CgpuCommandBuffer handle);
    void enqueueDestruction(CgpuBlas handle);
    void enqueueDestruction(CgpuTlas handle);

  private:
    using DestroyFunc = std::function<void()>;

    void enqueueDestroyFunc(DestroyFunc fun);

  private:
    CgpuDevice m_device;
    uint32_t m_frameIndex = 0;
    std::vector<DestroyFunc> m_pendingDestructions[FrameCount];
  };
}
