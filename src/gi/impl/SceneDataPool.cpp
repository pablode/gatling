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

#ifdef GI_TEST_EXECUTABLE
#include <doctest/doctest.h>

namespace
{
  using namespace gtl;

  struct TestFixture
  {
    CgpuContext* ctx = nullptr;
    std::unique_ptr<GgpuStager> stager;
    std::unique_ptr<GgpuDeleteQueue> deleteQueue;
    std::unique_ptr<GiSceneDataPool> pool;

    TestFixture()
    {
      ctx = cgpuCreateContext("gi_test");
      REQUIRE(ctx);

      stager = std::make_unique<GgpuStager>(ctx, /*hasSharedMem=*/false, /*size=*/1024 * 1024);
      REQUIRE(stager->allocate());

      deleteQueue = std::make_unique<GgpuDeleteQueue>(ctx);
      pool = std::make_unique<GiSceneDataPool>(ctx, *stager, *deleteQueue);
    }

    ~TestFixture()
    {
      pool->destroy();
      stager->flush();
      stager->free();
      deleteQueue->destroyAll();
      cgpuDestroyContext(ctx);
    }
  };

  std::vector<uint8_t> _MakeSceneData(uint8_t seed)
  {
    return std::vector<uint8_t>(64, seed);
  }
}

TEST_CASE_FIXTURE(TestFixture, "SceneDataPool.AcquireSameDataReturnsSameInstance")
{
  std::vector<uint8_t> data = _MakeSceneData(0x10);

  GiSceneDataPtr first = pool->acquire(data);
  REQUIRE(first);
  GiSceneDataPtr second = pool->acquire(data);
  REQUIRE(second);

  CHECK_EQ(first.get(), second.get());
  CHECK_EQ(first.use_count(), 2);
}

TEST_CASE_FIXTURE(TestFixture, "SceneDataPool.AcquireDifferentDataReturnsDifferentInstance")
{
  std::vector<uint8_t> firstData = _MakeSceneData(0x20);
  std::vector<uint8_t> secondData = _MakeSceneData(0x40);

  GiSceneDataPtr first = pool->acquire(firstData);
  REQUIRE(first);
  GiSceneDataPtr second = pool->acquire(secondData);
  REQUIRE(second);

  CHECK_NE(first.get(), second.get());
  CHECK_NE(first->buffer.handle, second->buffer.handle);
}

TEST_CASE_FIXTURE(TestFixture, "SceneDataPool.AcquireEmptyDataReturnsNull")
{
  std::vector<uint8_t> data;
  CHECK_FALSE(pool->acquire(data));
}

TEST_CASE_FIXTURE(TestFixture, "SceneDataPool.AcquiredBufferIsAligned")
{
  std::vector<uint8_t> data = _MakeSceneData(0x60);

  GiSceneDataPtr sceneData = pool->acquire(data);
  REQUIRE(sceneData);

  CHECK(sceneData->buffer.handle);
  CHECK_EQ(sceneData->address % rp::SCENE_DATA_ALIGNMENT, 0);
  CHECK_EQ(sceneData->size, data.size());
}

TEST_CASE_FIXTURE(TestFixture, "SceneDataPool.KeepsInstanceWhileReferenced")
{
  std::vector<uint8_t> data = _MakeSceneData(0x80);

  GiSceneDataPtr first = pool->acquire(data);
  REQUIRE(first);
  GiSceneDataPtr second = pool->acquire(data);
  REQUIRE(second);

  first.reset();

  GiSceneDataPtr third = pool->acquire(data);
  REQUIRE(third);

  CHECK_EQ(second.get(), third.get());
  CHECK_EQ(second.use_count(), 2);
}

TEST_CASE_FIXTURE(TestFixture, "SceneDataPool.ReleasesInstanceAfterLastReference")
{
  std::vector<uint8_t> data = _MakeSceneData(0xA0);

  GiSceneDataPtr first = pool->acquire(data);
  REQUIRE(first);

  CgpuBuffer firstBuffer = first->buffer;
  std::weak_ptr<GiSceneData> weakSceneData = first;

  first.reset();
  CHECK(weakSceneData.expired());

  GiSceneDataPtr second = pool->acquire(data);
  REQUIRE(second);
  CHECK_NE(second->buffer.handle, firstBuffer.handle); // first buffer is still alive in queue
}
#endif
