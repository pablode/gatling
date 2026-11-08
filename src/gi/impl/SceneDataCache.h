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

#include <stdint.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <gtl/cgpu/Cgpu.h>

#include <Gi.h>

namespace gtl
{
  class GgpuDeleteQueue;
  class GgpuStager;

  struct GiSceneData
  {
    std::vector<uint8_t> data;
    uint32_t uncompressedSize = 0;
    bool isCompressed = false;
    GiPrimvarInterpolation interpolation = GiPrimvarInterpolation::Constant;

    CgpuBuffer buffer;
    uint32_t addressIndex = UINT32_MAX;

    std::vector<uint8_t> decompress() const;
  };

  using GiSceneDataPtr = std::shared_ptr<GiSceneData>;

  class GiSceneDataCache
  {
  public:
    GiSceneDataCache(CgpuContext* ctx, GgpuStager& stager, GgpuDeleteQueue& deleteQueue);

    bool initialize();
    void destroy();

    GiSceneDataPtr addPrimvar(const GiPrimvarData& primvar,
                              const std::vector<GiFace>* faces,
                              uint64_t topologyHash);

    bool upload(const GiSceneDataPtr& sceneData);

    CgpuBuffer addressBuffer() const;

    void logStatsAndReset();

  private:
    struct CacheEntry
    {
      std::weak_ptr<GiSceneData> data;
      GiSceneData* raw = nullptr;
    };

    struct Cache
    {
      std::mutex mutex;
      std::unordered_map<uint64_t, CacheEntry> entries;
      std::vector<uint16_t> freeAddressIndices;
      uint32_t nextAddressIndex = 0;
      bool alive = true;
    };

  private:
    GiSceneDataPtr makeSceneDataPtr(GiSceneData* data, uint64_t hash);
    bool allocateAddressIndex(uint16_t& addressIndex);
    bool stageAddress(uint16_t addressIndex, uint64_t address);

    uint64_t makeKey(const GiPrimvarData& primvar, uint64_t topologyHash) const;

  private:
    CgpuContext* m_ctx;
    GgpuStager& m_stager;
    GgpuDeleteQueue& m_deleteQueue;
    CgpuBuffer m_addressBuffer;
    std::shared_ptr<Cache> m_cache;

    uint64_t m_cpuReferenceCount = 0;
    uint64_t m_cpuUniqueCount = 0;
    uint64_t m_cpuHitCount = 0;
    uint64_t m_cpuRawBytes = 0;
    uint64_t m_cpuOptimizedBytes = 0;
    uint64_t m_uploadedBytes = 0;
  };
}
