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

#include "SyncBuffer.h"
#include "Stager.h"

#include <algorithm>

namespace gtl
{
  GgpuSyncBuffer::GgpuSyncBuffer(CgpuDevice device,
                                 GgpuStager& stager,
                                 GgpuDelayedResourceDestroyer& delayedResourceDestroyer,
                                 uint64_t elementSize,
                                 CgpuBufferUsage bufferUsage)
    : m_device(device)
    , m_stager(stager)
    , m_elementSize(elementSize)
    , m_deviceBuffer(m_device,
                     delayedResourceDestroyer,
                     bufferUsage | CgpuBufferUsage::TransferDst,
                     CgpuMemoryProperties::DeviceLocal)
    , m_hostBuffer(m_device,
                   delayedResourceDestroyer,
                   CgpuBufferUsage::TransferSrc,
                   CgpuMemoryProperties::HostVisible | CgpuMemoryProperties::HostCoherent)
  {
  }

  GgpuSyncBuffer::~GgpuSyncBuffer()
  {
    if (m_mappedHostMem)
    {
      cgpuUnmapBuffer(m_device, m_hostBuffer.buffer());
      m_mappedHostMem = nullptr;
    }
  }

  uint8_t* GgpuSyncBuffer::read(uint64_t byteOffset, uint64_t byteSize)
  {
    assert((byteOffset + byteSize) <= m_size);
    return &m_mappedHostMem[byteOffset];
  }

  uint8_t* GgpuSyncBuffer::write(uint64_t byteOffset, uint64_t byteSize)
  {
    uint64_t rangeEnd = byteOffset + byteSize;
    assert(rangeEnd <= m_size);
    m_dirtyRangeBegin = std::min(byteOffset, m_dirtyRangeBegin);
    m_dirtyRangeEnd = std::max(rangeEnd, m_dirtyRangeEnd);
    return &m_mappedHostMem[byteOffset];
  }

  CgpuBuffer GgpuSyncBuffer::buffer() const
  {
    return m_deviceBuffer.buffer();
  }

  uint64_t GgpuSyncBuffer::byteSize() const
  {
    return m_size;
  }

  bool GgpuSyncBuffer::resize(CgpuDevice device, CgpuCommandBuffer commandBuffer, uint64_t newSize)
  {
    if (newSize == m_size)
    {
      assert(false); // Ideally resize() should only be called if its needed
      return true;
    }

    // Unmap buffer before resize. New buffer is mapped later.
    if (m_mappedHostMem)
    {
      cgpuUnmapBuffer(device, m_hostBuffer.buffer());
      m_mappedHostMem = nullptr;
    }

    m_size = newSize;

    // Reset buffers if new size is 0.
    if (newSize == 0)
    {
      m_hostBuffer.resize(0, commandBuffer);
      m_deviceBuffer.resize(0, commandBuffer);
      return true;
    }

    // Resize buffers.
    m_deviceBuffer.resize(newSize, commandBuffer);

    m_hostBuffer.resize(newSize, commandBuffer);

    cgpuMapBuffer(device, m_hostBuffer.buffer(), (void**) &m_mappedHostMem);

    return true;
  }

  bool GgpuSyncBuffer::commitChanges()
  {
    if (m_dirtyRangeBegin == UINT64_MAX && m_dirtyRangeEnd == 0)
    {
      // Nothing to commit.
      return true;
    }

    if (m_dirtyRangeBegin > m_dirtyRangeEnd)
    {
      assert(false);
      return false;
    }

    // Vulkan spec conformance: offset and size must be multiples of 4.
    m_dirtyRangeBegin = (m_dirtyRangeBegin < 4) ? 0 : (m_dirtyRangeBegin / 4) * 4;
    m_dirtyRangeEnd = (m_dirtyRangeEnd + 3) / 4 * 4;

    uint64_t commitSize = m_dirtyRangeEnd - m_dirtyRangeBegin;

    if (commitSize == 0)
    {
      assert(false);
      return true;
    }

    if (!m_stager.stageToBuffer(&m_mappedHostMem[m_dirtyRangeBegin],
                                commitSize,
                                m_deviceBuffer.buffer(),
                                m_dirtyRangeBegin))
    {
      return false;
    }

    m_dirtyRangeBegin = UINT64_MAX;
    m_dirtyRangeEnd = 0;

    return true;
  }
}
