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

#include "DeleteQueue.h"

#include <assert.h>

namespace gtl
{
  GgpuDeleteQueue::GgpuDeleteQueue(CgpuContext* ctx)
    : m_ctx(ctx)
  {
  }

  GgpuDeleteQueue::~GgpuDeleteQueue()
  {
    for (uint32_t i = 0; i < FrameCount; i++)
    {
      assert(m_pendingDestructions[i].empty());
    }
  }

  void GgpuDeleteQueue::housekeep()
  {
    auto& oldestFrameDestructions = m_pendingDestructions[m_frameIndex];
    for (const DestroyFunc& fun : oldestFrameDestructions)
    {
      fun();
    }

    oldestFrameDestructions.clear();
  }

  void GgpuDeleteQueue::nextFrame()
  {
    m_frameIndex = (m_frameIndex + 1) % FrameCount;
  }

  void GgpuDeleteQueue::destroyAll()
  {
    for (uint32_t i = 0; i < FrameCount; i++)
    {
      nextFrame();
      housekeep();
    }
  }

  void GgpuDeleteQueue::pushBack(CgpuBuffer handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([this, handle]() { cgpuDestroyBuffer(m_ctx, handle); });
  }

  void GgpuDeleteQueue::pushBack(CgpuImage handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([this, handle]() { cgpuDestroyImage(m_ctx, handle); });
  }

  void GgpuDeleteQueue::pushBack(CgpuPipeline handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([this, handle]() { cgpuDestroyPipeline(m_ctx, handle); });
  }

  void GgpuDeleteQueue::pushBack(CgpuSemaphore handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([this, handle]() { cgpuDestroySemaphore(m_ctx, handle); });

  void GgpuDeleteQueue::pushBack(CgpuCommandBuffer handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([this, handle]() { cgpuDestroyCommandBuffer(m_ctx, handle); });
  }

  void GgpuDeleteQueue::pushBack(CgpuBlas handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([this, handle]() { cgpuDestroyBlas(m_ctx, handle); });
  }

  void GgpuDeleteQueue::pushBack(CgpuTlas handle)
  {
    if (!handle.handle) { return; }
    enqueueDestroyFunc([this, handle]() { cgpuDestroyTlas(m_ctx, handle); });
  }

  void GgpuDeleteQueue::enqueueDestroyFunc(DestroyFunc fun)
  {
    m_pendingDestructions[m_frameIndex].push_back(fun);
  }
}
