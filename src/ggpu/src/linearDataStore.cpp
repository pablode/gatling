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

#include "linearDataStore.h"

namespace
{
  // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  uint32_t _NextPowerOfTwo(uint32_t v)
  {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
  }
}

namespace gtl
{
  GgpuLinearDataStore::GgpuLinearDataStore(CgpuDevice device,
                                           GgpuStager& stager,
                                           uint64_t elementSize, 
                                           uint32_t minCapacity)
    : m_device(device)
    , m_elementSize(elementSize)
    , m_minCapacity(minCapacity)
    , m_elementCount(0)
    , m_buffer(device, stager, elementSize)
  {
  }

  GgpuLinearDataStore::~GgpuLinearDataStore()
  {
    if (m_buffer.byteSize() > 0)
    {
      CgpuCommandBuffer commandBuffer; // TODO
      m_buffer.resize(m_device, commandBuffer, 0);
    }
  }

  uint64_t GgpuLinearDataStore::allocate()
  {
    m_elementCount++;
    return m_handleStore.allocateHandle();
  }

  void GgpuLinearDataStore::free(uint64_t handle)
  {
    m_elementCount--;
    m_handleStore.freeHandle(handle);
  }

  uint8_t* GgpuLinearDataStore::readRaw(uint64_t handle)
  {
    uint64_t byteOffset = returnOrAllocHandle(handle);
    return (byteOffset == UINT64_MAX) ? nullptr : m_buffer.read(byteOffset, m_elementSize);
  }

  uint8_t* GgpuLinearDataStore::writeRaw(uint64_t handle)
  {
    uint64_t byteOffset = returnOrAllocHandle(handle);
    return (byteOffset == UINT64_MAX) ? nullptr : m_buffer.write(byteOffset, m_elementSize);
  }

  uint8_t* GgpuLinearDataStore::readFromIndex(uint32_t index)
  {
    uint64_t byteOffset = returnOrAllocIndex(index);
    return (byteOffset == UINT64_MAX) ? nullptr : m_buffer.read(byteOffset, m_elementSize);
  }

  uint8_t* GgpuLinearDataStore::writeToIndex(uint32_t index)
  {
    uint64_t byteOffset = returnOrAllocIndex(index);
    return (byteOffset == UINT64_MAX) ? nullptr : m_buffer.write(byteOffset, m_elementSize);
  }

  uint64_t GgpuLinearDataStore::returnOrAllocHandle(uint64_t handle)
  {
    if (!m_handleStore.isHandleValid(handle))
    {
      assert(false);
      return UINT32_MAX;
    }

    uint32_t index = uint32_t(handle);
    return returnOrAllocIndex(index);
  }

  uint64_t GgpuLinearDataStore::returnOrAllocIndex(uint32_t index)
  {
    uint64_t byteOffset = index * m_elementSize;

    if (byteOffset >= m_buffer.byteSize())
    {
      uint32_t minSize = m_elementSize * m_minCapacity;
      uint32_t newSize = std::max(_NextPowerOfTwo(byteOffset), minSize);

      CgpuCommandBuffer commandBuffer; // TODO
      if (!m_buffer.resize(m_device, commandBuffer, newSize))
      {
        assert(false);
        return UINT32_MAX;
      }
    }

    return byteOffset;
  }

  CgpuBuffer GgpuLinearDataStore::buffer() const
  {
    return m_buffer.buffer();
  }

  uint64_t GgpuLinearDataStore::bufferSize() const
  {
    return m_buffer.byteSize();
  }

  bool GgpuLinearDataStore::commitChanges()
  {
    return m_buffer.commitChanges();
  }

  uint32_t GgpuLinearDataStore::elementCount() const
  {
    return m_elementCount;
  }
}

#ifdef TEST_EXECUTABLE
#include <doctest/doctest.h>
#include "stager.h"
#include <memory>

using namespace gtl;

class GgpuTestFixture {
protected:
  CgpuDevice m_device;
public:
  GgpuTestFixture()
  {
    REQUIRE(cgpuInitialize("ggpu_test", 1, 0, 0));
    REQUIRE(cgpuCreateDevice(&m_device));
  }
  ~GgpuTestFixture()
  {
    REQUIRE(cgpuDestroyDevice(m_device));
    cgpuTerminate();
  }
};

TEST_CASE_FIXTURE(GgpuTestFixture, "SimpleWriteRead")
{
  struct TestElement {
    uint32_t e0 = 0x40;
    uint32_t e1 = 0x80;
  } testElement1;

  GgpuStager stager(m_device);
  REQUIRE(stager.allocate());
  GgpuLinearDataStore dataStore(m_device, stager, sizeof(TestElement), 1);

  uint64_t handle = dataStore.allocate();
  *dataStore.write<TestElement>(handle) = testElement1;
  REQUIRE(dataStore.commitChanges());
  REQUIRE(dataStore.elementCount() == 1);

  TestElement testElement2 = *dataStore.read<TestElement>(handle);
  CHECK(testElement2.e0 == testElement1.e0);
  CHECK(testElement2.e1 == testElement1.e1);

  dataStore.free(handle);
  stager.free();
}
#endif
