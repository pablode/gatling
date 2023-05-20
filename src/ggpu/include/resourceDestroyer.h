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

#include <functional>
#include <vector>
#include <list>

namespace gtl
{
  class GgpuResourceDestroyer
  {
  public:
    GgpuResourceDestroyer(CgpuDevice device);

    ~GgpuResourceDestroyer();

  public:
    using DestroyFunc = std::function<bool(CgpuDevice device)>;

    void enqueueDestruction(DestroyFunc callback);

    CgpuFence getFenceWithDestructionCallback(DestroyFunc callback);

  public:
    void destroyUnusedResources();

    void destroyAllResources();

  private:
    struct FencedCallback
    {
      CgpuFence fence;
      DestroyFunc callback;
    };

    CgpuDevice m_device;
    std::list<FencedCallback> m_pendingDestructions;
    std::vector<CgpuFence> m_fenceFreeList;
  };
}
