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

#include "SceneDataCache.h"

#include "MeshProcessing.h"

#include <gtl/gb/Fmt.h>
#include <gtl/gb/Hash.h>
#include <gtl/gb/Log.h>
#include <gtl/ggpu/DeleteQueue.h>
#include <gtl/ggpu/Stager.h>

#include <blosc2.h>
#include <xxhash.h>

#include <string.h>

#include "interface/rp_main.h"

namespace
{
  constexpr uint32_t COMPRESSION_THRESHOLD = 1024;

  std::vector<uint8_t> _CompressData(const uint8_t* data, uint64_t size, uint32_t typeSize)
  {
    std::vector<uint8_t> compressed(size + BLOSC2_MAX_OVERHEAD);

    int compressedSize = blosc1_compress(3, BLOSC_BITSHUFFLE, typeSize, size, data, compressed.data(), compressed.size());
    if (compressedSize <= 0)
    {
      return {};
    }

    compressed.resize(compressedSize);
    compressed.shrink_to_fit();
    return compressed;
  }
}

namespace gtl
{
  namespace rp = shader_interface::rp_main;

  std::vector<uint8_t> GiSceneData::decompress() const
  {
    if (!isCompressed)
    {
      return data;
    }

    std::vector<uint8_t> uncompressed(uncompressedSize);
    int result = blosc1_decompress(data.data(), uncompressed.data(), uncompressed.size());
    if (result < 0 || uint32_t(result) != uncompressedSize)
    {
      GB_ERROR("failed to decompress scene data");
      return {};
    }
    return uncompressed;
  }

  GiSceneDataCache::GiSceneDataCache(CgpuContext* ctx,
                                     GgpuStager& stager,
                                     GgpuDeleteQueue& deleteQueue)
    : m_ctx(ctx)
    , m_stager(stager)
    , m_deleteQueue(deleteQueue)
  {
    m_cache = std::make_shared<Cache>();
  }

  bool GiSceneDataCache::initialize()
  {
    uint64_t addressBufferSize = uint64_t(rp::MAX_SCENE_DATA_ADDRESS_COUNT) * sizeof(uint64_t);

    if (!cgpuCreateBuffer(m_ctx, {
                            .usage = CgpuBufferUsage::Storage | CgpuBufferUsage::TransferDst,
                            .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                            .size = addressBufferSize,
                            .debugName = "SceneDataAddresses"
                          }, &m_addressBuffer))
    {
      GB_ERROR("failed to create scene data address buffer");
      return false;
    }

    return true;
  }

  void GiSceneDataCache::destroy()
  {
    std::lock_guard guard(m_cache->mutex);
    m_cache->alive = false;

    for (auto& entry : m_cache->entries)
    {
      if (GiSceneDataPtr data = entry.second.data.lock(); data)
      {
        if (data->buffer.handle)
        {
          cgpuDestroyBuffer(m_ctx, data->buffer);
        }
        data->buffer = {};
        data->addressIndex = UINT32_MAX;
      }
    }

    m_cache->entries.clear();
    m_cache->freeAddressIndices.clear();

    if (m_addressBuffer.handle)
    {
      cgpuDestroyBuffer(m_ctx, m_addressBuffer);
      m_addressBuffer = {};
    }
  }

  GiSceneDataPtr GiSceneDataCache::addPrimvar(const GiPrimvarData& primvar,
                                              const std::vector<GiFace>* faces,
                                              uint64_t topologyHash)
  {
    uint32_t typeSize = giPrimvarTypeSize(primvar.type);
    if (typeSize == 0 || primvar.data.empty() || (primvar.data.size() % typeSize) != 0)
    {
      GB_ERROR("invalid scene data for {}", primvar.name);
      return nullptr;
    }

    uint64_t key = makeKey(primvar, topologyHash);

    {
      std::lock_guard guard(m_cache->mutex);
      if (!m_cache->alive)
      {
        return nullptr;
      }

      m_cpuReferenceCount++;
      m_cpuRawBytes += primvar.data.size();

      auto cacheIt = m_cache->entries.find(key);
      if (cacheIt != m_cache->entries.end())
      {
        if (GiSceneDataPtr sceneData = cacheIt->second.data.lock(); sceneData)
        {
          m_cpuHitCount++;
          return sceneData;
        }
        m_cache->entries.erase(cacheIt);
      }
    }

    GiPrimvarInterpolation interpolation = primvar.interpolation;
    const uint8_t* data = primvar.data.data();
    uint64_t size = primvar.data.size();
    std::vector<uint8_t> optimizedData;

    if (interpolation == GiPrimvarInterpolation::Constant)
    {
      size = typeSize;
    }
    else if (giPrimvarCanBeConstant(primvar.data, typeSize))
    {
      GB_DEBUG("scene data {} converted to constant interpolation", primvar.name);
      interpolation = GiPrimvarInterpolation::Constant;
      size = typeSize;
    }
    else if (faces && !faces->empty() &&
             (interpolation == GiPrimvarInterpolation::Vertex || interpolation == GiPrimvarInterpolation::FaceVarying) &&
             giPrimvarCanBeUniform(interpolation, *faces, primvar.data, typeSize))
    {
      optimizedData.resize(faces->size() * typeSize);
      for (size_t i = 0; i < faces->size(); i++)
      {
        uint32_t sourceIndex = (interpolation == GiPrimvarInterpolation::Vertex) ? (*faces)[i].v_i[0] : uint32_t(i * 3);
        memcpy(&optimizedData[i * typeSize], &primvar.data[sourceIndex * typeSize], typeSize);
      }

      GB_DEBUG("scene data {} converted to uniform interpolation", primvar.name);
      interpolation = GiPrimvarInterpolation::Uniform;
      data = optimizedData.data();
      size = optimizedData.size();
    }

    GiSceneData* rawEntry = new GiSceneData;
    rawEntry->interpolation = interpolation;
    rawEntry->uncompressedSize = uint32_t(size);

    if (size >= COMPRESSION_THRESHOLD)
    {
      std::vector<uint8_t> compressed = _CompressData(data, size, typeSize);
      if (!compressed.empty() && compressed.size() < size)
      {
        rawEntry->data = std::move(compressed);
        rawEntry->isCompressed = true;
      }
    }
    if (!rawEntry->isCompressed)
    {
      rawEntry->data.assign(data, data + size);
    }

    GiSceneDataPtr sceneData = makeSceneDataPtr(rawEntry, key);

    {
      std::lock_guard guard(m_cache->mutex);

      if (!m_cache->alive)
      {
        return nullptr;
      }

      auto cacheIt = m_cache->entries.find(key);
      if (cacheIt != m_cache->entries.end())
      {
        if (GiSceneDataPtr existing = cacheIt->second.data.lock(); existing)
        {
          m_cpuHitCount++;
          return existing;
        }
        m_cache->entries.erase(cacheIt);
      }

      m_cache->entries[key] = { .data = sceneData, .raw = rawEntry };
      m_cpuUniqueCount++;
      m_cpuOptimizedBytes += rawEntry->data.size();
    }

    return sceneData;
  }

  bool GiSceneDataCache::upload(const GiSceneDataPtr& sceneData)
  {
    if (!sceneData)
    {
      return false;
    }

    {
      std::lock_guard guard(m_cache->mutex);
      if (!m_cache->alive)
      {
        return false;
      }

      if (sceneData->addressIndex != UINT32_MAX)
      {
        return true;
      }
    }

    std::vector<uint8_t> data = sceneData->decompress();
    if (data.size() != sceneData->uncompressedSize)
    {
      GB_ERROR("failed to decompress scene data");
      return false;
    }

    std::string debugName = GB_FMT("SceneData_{}", uintptr_t(sceneData.get()));

    CgpuBuffer buffer;
    if (!cgpuCreateBuffer(m_ctx, {
                            .usage = CgpuBufferUsage::ShaderDeviceAddress | CgpuBufferUsage::TransferDst,
                            .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                            .size = data.size(),
                            .debugName = debugName.c_str()
                          }, &buffer))
    {
      GB_ERROR("failed to create scene data buffer");
      return false;
    }

    uint64_t address = cgpuGetBufferGpuAddress(m_ctx, buffer);
    if (address == 0)
    {
      GB_ERROR("failed to get scene data buffer address");
      cgpuDestroyBuffer(m_ctx, buffer);
      return false;
    }

    uint16_t addressIndex;
    if (!allocateAddressIndex(addressIndex))
    {
      cgpuDestroyBuffer(m_ctx, buffer);
      return false;
    }

    if (!m_stager.stageToBuffer(data.data(), data.size(), buffer) ||
        !stageAddress(addressIndex, address))
    {
      GB_ERROR("failed to stage scene data");
      cgpuDestroyBuffer(m_ctx, buffer);
      return false;
    }

    sceneData->buffer = buffer;
    sceneData->addressIndex = addressIndex;

    {
      std::lock_guard guard(m_cache->mutex);
      m_uploadedBytes += data.size();
    }

    return true;
  }

  CgpuBuffer GiSceneDataCache::addressBuffer() const
  {
    return m_addressBuffer;
  }

  void GiSceneDataCache::logStatsAndReset()
  {
    constexpr float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

    if (m_cpuReferenceCount > 0)
    {
      GB_LOG("scene data: {} unique of {} references ({} cache hits, {:.2f} MiB raw, {:.2f} MiB optimized, {:.2f} MiB uploaded)",
             m_cpuUniqueCount, m_cpuReferenceCount, m_cpuHitCount,
             float(m_cpuRawBytes) * BYTES_TO_MIB, float(m_cpuOptimizedBytes) * BYTES_TO_MIB,
             float(m_uploadedBytes) * BYTES_TO_MIB);
    }

    m_cpuReferenceCount = 0;
    m_cpuUniqueCount = 0;
    m_cpuHitCount = 0;
    m_cpuRawBytes = 0;
    m_cpuOptimizedBytes = 0;
    m_uploadedBytes = 0;
  }

  GiSceneDataPtr GiSceneDataCache::makeSceneDataPtr(GiSceneData* data, uint64_t hash)
  {
    std::shared_ptr<Cache> cache = m_cache;
    GgpuDeleteQueue* deleteQueue = &m_deleteQueue;

    return GiSceneDataPtr(data, [cache, deleteQueue, hash](GiSceneData* sceneData)
                          {
                            std::lock_guard guard(cache->mutex);

                            auto cacheIt = cache->entries.find(hash);
                            if (cacheIt != cache->entries.end() && cacheIt->second.raw == sceneData)
                            {
                              cache->entries.erase(cacheIt);
                              GB_DEBUG("scene data unloaded ({} bytes)",
                                       sceneData->isCompressed ? sceneData->data.size() : sceneData->uncompressedSize);
                            }

                            if (cache->alive && sceneData->buffer.handle)
                            {
                              deleteQueue->pushBack(sceneData->buffer);
                            }

                            if (sceneData->addressIndex != UINT32_MAX)
                            {
                              cache->freeAddressIndices.push_back(uint16_t(sceneData->addressIndex));
                            }

                            delete sceneData;
                          });
  }

  uint64_t GiSceneDataCache::makeKey(const GiPrimvarData& primvar, uint64_t topologyHash) const
  {
    GbHash hash = GbHashAppend(GbHash{}, uint32_t(primvar.type));
    hash = GbHashAppend(hash, uint32_t(primvar.interpolation));

    if (primvar.interpolation == GiPrimvarInterpolation::Vertex || primvar.interpolation == GiPrimvarInterpolation::FaceVarying)
    {
      hash = GbHashAppend(hash, topologyHash);
    }

    hash = GbHashAppend(hash, XXH64(primvar.data.data(), primvar.data.size(), 0));

    return hash.val;
  }

  bool GiSceneDataCache::allocateAddressIndex(uint16_t& addressIndex)
  {
    std::lock_guard guard(m_cache->mutex);

    if (!m_cache->freeAddressIndices.empty())
    {
      addressIndex = m_cache->freeAddressIndices.back();
      m_cache->freeAddressIndices.pop_back();
      return true;
    }

    if (m_cache->nextAddressIndex < rp::MAX_SCENE_DATA_ADDRESS_COUNT)
    {
      addressIndex = uint16_t(m_cache->nextAddressIndex++);
      return true;
    }

    GB_ERROR("scene data address buffer exhausted");
    return false;
  }

  bool GiSceneDataCache::stageAddress(uint16_t addressIndex, uint64_t address)
  {
    uint64_t offset = uint64_t(addressIndex) * sizeof(address);
    return m_stager.stageToBuffer((const uint8_t*) &address, sizeof(address), m_addressBuffer, offset);
  }
}
