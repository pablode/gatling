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

#include "resourceDestroyer.h"

#include <assert.h>

namespace gtl
{
  GgpuResourceDestroyer::GgpuResourceDestroyer(CgpuDevice device)
    : m_device(device)
  {
  }

  GgpuResourceDestroyer::~GgpuResourceDestroyer()
  {
    assert(m_pendingDestructions.empty());
    assert(m_fenceFreeList.empty());
  }

  void GgpuResourceDestroyer::enqueueDestruction(DestroyFunc callback)
  {
    CgpuFence invalidFence = { CGPU_INVALID_HANDLE };
    m_pendingDestructions.push_back(FencedCallback{ invalidFence, callback });
  }

  CgpuFence GgpuResourceDestroyer::getFenceWithDestructionCallback(DestroyFunc callback)
  {
    CgpuFence f = { CGPU_INVALID_HANDLE };

    if (m_fenceFreeList.empty())
    {
      if (!cgpuCreateFence(m_device, &f))
      {
        assert(false);
        return f;
      }
    }
    else
    {
      f = m_fenceFreeList.back();
      m_fenceFreeList.pop_back();
    }

    if (!cgpuResetFence(m_device, f))
    {
      assert(false);
    }

    m_pendingDestructions.push_back(FencedCallback{ f, callback });
    return f;
  }

  void GgpuResourceDestroyer::destroyUnusedResources()
  {
    auto it = m_pendingDestructions.begin();

    while (it != m_pendingDestructions.end())
    {
      CgpuFence fence = it->fence;
      DestroyFunc callback = it->callback;

      bool fenceValid = (fence.handle != CGPU_INVALID_HANDLE);

      if (fenceValid)
      {
        bool signalled = false;
        if (!cgpuGetFenceSignalled(m_device, fence, &signalled))
        {
          assert(false);
        }

        if (!signalled)
        {
          ++it;
          continue;
        }

        m_fenceFreeList.push_back(fence);
      }

      callback(m_device);

      it = m_pendingDestructions.erase(it);
    }
  }

  void GgpuResourceDestroyer::destroyAllResources()
  {
    for (const FencedCallback& fc : m_pendingDestructions)
    {
      CgpuFence f = fc.fence;

      if (f.handle == CGPU_INVALID_HANDLE)
      {
        continue;
      }

      cgpuWaitForFence(m_device, f);
    }

    destroyUnusedResources();
    assert(m_pendingDestructions.size() == 0);

    for (CgpuFence f : m_fenceFreeList)
    {
      cgpuDestroyFence(m_device, f);
    }
  }
}
