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

  void GgpuDelayedResourceDestroyer::nextFrame()
  {
    m_frameIndex = (m_frameIndex + 1) % FrameCount;

    auto& oldestFrameDestructions = m_pendingDestructions[m_frameIndex];
    for (const DestroyFunc& fun : oldestFrameDestructions)
    {
      fun();
    }

    oldestFrameDestructions.clear();
  }

  void GgpuDelayedResourceDestroyer::destroyAll()
  {
    for (uint32_t i = 0; i < FrameCount; i++)
    {
      nextFrame();
    }
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuBuffer handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([=]() { cgpuDestroyBuffer(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuImage handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([=]() { cgpuDestroyImage(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuPipeline handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([=]() { cgpuDestroyPipeline(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuSemaphore handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([=]() { cgpuDestroySemaphore(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuCommandBuffer handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([=]() { cgpuDestroyCommandBuffer(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuBlas handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([=]() { cgpuDestroyBlas(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestruction(CgpuTlas handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([=]() { cgpuDestroyTlas(m_device, handle); });
  }

  void GgpuDelayedResourceDestroyer::enqueueDestroyFunc(DestroyFunc fun)
  {
    m_pendingDestructions[m_frameIndex].push_back(fun);
  }
}
