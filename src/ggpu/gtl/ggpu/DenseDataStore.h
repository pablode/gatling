//
// Copyright (C) 2023 Pablo Delgado Krämer
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

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <unordered_map>

#include "LinearDataStore.h"

namespace gtl
{
  class GgpuDelayedResourceDestroyer;

  class GgpuDenseDataStore : public GgpuLinearDataStore
  {
  public:
    GgpuDenseDataStore(CgpuDevice device,
                       GgpuStager& stager,
                       GgpuDelayedResourceDestroyer& delayedResourceDestroyer,
                       uint64_t elementSize,
                       uint32_t minCapacity);

  public:
    uint64_t allocate() override;

    void free(uint64_t handle) override;

  protected:
    uint8_t* readRaw(uint64_t handle) override;

    uint8_t* writeRaw(uint64_t handle) override;

  private:
    std::unordered_map<uint64_t/*handle*/, uint32_t/*index*/> m_indexMap;
    uint64_t m_elementSize;
    uint32_t m_highestIndex = 0;
  };
}
