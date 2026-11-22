//
// Copyright (C) 2026 Pablo Delgado Krämer
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

#include <stdint.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <gtl/cgpu/Cgpu.h>

namespace gtl
{
  class GgpuDeleteQueue;
  class GgpuStager;

  struct GiSceneData
  {
    CgpuBuffer buffer;
    uint64_t address = 0;
    uint64_t size = 0;
  };

  using GiSceneDataPtr = std::shared_ptr<GiSceneData>;

  class GiSceneDataPool
  {
  public:
    GiSceneDataPool(CgpuContext* gpuCtx,
                    GgpuStager& stager,
                    GgpuDeleteQueue& deleteQueue);

    void destroy();

    GiSceneDataPtr acquire(const std::vector<uint8_t>& data);

    void logStats();

  private:
    GiSceneDataPtr makeSceneDataPtr(GiSceneData* sceneData, uint64_t hash);

  private:
    CgpuContext* m_gpuCtx;
    GgpuStager& m_stager;
    GgpuDeleteQueue& m_deleteQueue;
    std::mutex m_mutex;
    std::unordered_map<uint64_t, std::weak_ptr<GiSceneData>> m_cache;
    uint64_t m_bytesStored = 0;
    uint32_t m_sceneDataCount = 0;
  };
}
