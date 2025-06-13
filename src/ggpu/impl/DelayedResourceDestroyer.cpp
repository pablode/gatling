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

#include "DelayedResourceDestroyer.h"

#include <assert.h>

namespace gtl
{
  GgpuDelayedResourceDestroyer::GgpuDelayedResourceDestroyer(CgpuDevice device)
    : m_device(device)
  {
  }

  GgpuDelayedResourceDestroyer::~GgpuDelayedResourceDestroyer()
  {
    for (uint32_t i = 0; i < FrameCount; i++)
    {
      assert(m_pendingDestructions[i].empty());
    }
  }

  void GgpuDelayedResourceDestroyer::housekeep()
  {
    auto& oldestFrameDestructions = m_pendingDestructions[m_frameIndex];
    for (const DestroyFunc& fun : oldestFrameDestructions)
    {
      fun();
    }

    oldestFrameDestructions.clear();
  }

  void GgpuDelayedResourceDestroyer::nextFrame()
  {
    m_frameIndex = (m_frameIndex + 1) % FrameCount;
  }

  void GgpuDelayedResourceDestroyer::destroyAll()
  {
    for (uint32_t i = 0; i < FrameCount; i++)
    {
      nextFrame();
      housekeep();
    }
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuBuffer handle)
  {
    assert(handle.handle);
    enqueueDestroyFunc([=]() { cgpuDestroyBuffer(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuImage handle)
  {
    assert(handle.handle);
    enqueueDestroyFunc([=]() { cgpuDestroyImage(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuPipeline handle)
  {
    assert(handle.handle);
    enqueueDestroyFunc([=]() { cgpuDestroyPipeline(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuSemaphore handle)
  {
    assert(handle.handle);
    enqueueDestroyFunc([=]() { cgpuDestroySemaphore(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuCommandBuffer handle)
  {
    assert(handle.handle);
    enqueueDestroyFunc([=]() { cgpuDestroyCommandBuffer(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuBlas handle)
  {
    assert(handle.handle);
    enqueueDestroyFunc([=]() { cgpuDestroyBlas(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuTlas handle)
  {
    assert(handle.handle);
    enqueueDestroyFunc([=]() { cgpuDestroyTlas(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestroyFunc(DestroyFunc fun)
  {
    m_pendingDestructions[m_frameIndex].push_back(fun);
  }
}
