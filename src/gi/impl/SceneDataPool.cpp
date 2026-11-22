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

#include "SceneDataPool.h"

#include "interface/rp_main.h"

#include <gtl/gb/Fmt.h>
#include <gtl/gb/Log.h>
#include <gtl/ggpu/DeleteQueue.h>
#include <gtl/ggpu/Stager.h>

#include <xxhash.h>

#include <stdlib.h>

namespace
{
  bool _IsPoolingDisabled()
  {
    return getenv("GTL_DISABLE_SCENE_DATA_POOLING") != nullptr;
  }
}

namespace gtl
{
  namespace rp = shader_interface::rp_main;

  GiSceneDataPool::GiSceneDataPool(CgpuContext* gpuCtx,
                                   GgpuStager& stager,
                                   GgpuDeleteQueue& deleteQueue)
    : m_gpuCtx(gpuCtx)
    , m_stager(stager)
    , m_deleteQueue(deleteQueue)
  {
  }

  void GiSceneDataPool::destroy()
  {
    std::lock_guard guard(m_mutex);
    m_cache.clear();
  }

  GiSceneDataPtr GiSceneDataPool::acquire(const std::vector<uint8_t>& data)
  {
    if (data.empty())
    {
      return nullptr;
    }

    uint64_t hash = XXH64(data.data(), data.size(), 0);

    if (!_IsPoolingDisabled())
    {
      std::lock_guard guard(m_mutex);

      auto cacheIt = m_cache.find(hash);
      if (cacheIt != m_cache.end())
      {
        if (GiSceneDataPtr sceneData = cacheIt->second.lock(); sceneData)
        {
          return sceneData;
        }
      }
    }

    std::string debugName = GB_FMT("SceneData_{}", hash);

    uint64_t bufferSize = data.size() + rp::SCENE_DATA_ALIGNMENT - 1;

    CgpuBuffer buffer;
    if (!cgpuCreateBuffer(m_gpuCtx, {
                            .usage = CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::TransferDst,
                            .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                            .size = bufferSize,
                            .debugName = debugName.c_str(),
                            .alignment = uint32_t(rp::SCENE_DATA_ALIGNMENT)
                          }, &buffer))
    {
      GB_ERROR("failed to create scene data buffer");
      return nullptr;
    }

    uint64_t bufferAddress = cgpuGetBufferGpuAddress(m_gpuCtx, buffer);
    uint64_t address = (bufferAddress + rp::SCENE_DATA_ALIGNMENT - 1) & ~(rp::SCENE_DATA_ALIGNMENT - 1);
    uint64_t dataOffset = address - bufferAddress;

    if (!m_stager.stageToBuffer(data.data(), data.size(), buffer, dataOffset))
    {
      GB_ERROR("failed to upload scene data");
      cgpuDestroyBuffer(m_gpuCtx, buffer);
      return nullptr;
    }

    GiSceneDataPtr sceneData = makeSceneDataPtr(new GiSceneData {
      .buffer = buffer,
      .address = address,
      .size = data.size()
    }, hash);

    {
      std::lock_guard guard(m_mutex);

      if (!_IsPoolingDisabled())
      {
        m_cache[hash] = sceneData;
      }

      m_bytesStored += data.size();
      m_sceneDataCount++;
    }

    return sceneData;
  }

  GiSceneDataPtr GiSceneDataPool::makeSceneDataPtr(GiSceneData* sceneData, uint64_t hash)
  {
    return GiSceneDataPtr(sceneData, [this, hash](GiSceneData* data) {
      {
        std::lock_guard guard(m_mutex);

        if (!_IsPoolingDisabled())
        {
          auto cacheIt = m_cache.find(hash);
          if (cacheIt != m_cache.end() && cacheIt->second.expired())
          {
            m_cache.erase(cacheIt);
          }
        }

        m_bytesStored -= data->size;
        m_sceneDataCount--;
      }

      if (data->buffer.handle)
      {
        m_deleteQueue.pushBack(data->buffer);
      }

      delete data;
    });
  }

  void GiSceneDataPool::logStats()
  {
    constexpr double BYTES_TO_MIB = 1.0 / (1024.0 * 1024.0);

    std::lock_guard guard(m_mutex);

    GB_LOG("Scene data pool");
    GB_LOG("> {} unique buffers", m_sceneDataCount);
    GB_LOG("> {:.2f} MiB", double(m_bytesStored) * BYTES_TO_MIB);
  }
}
