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

#include "BumpAllocator.h"
#include "DelayedResourceDestroyer.h"

#include <assert.h>

namespace gtl
{
  std::shared_ptr<GgpuBumpAllocator> GgpuBumpAllocator::make(CgpuContext* ctx,
                                                             GgpuDelayedResourceDestroyer& delayedResourceDestroyer,
                                                             uint32_t size)
  {
    CgpuBuffer buffer;
    if (!cgpuCreateBuffer(ctx, { .usage = CgpuBufferUsage::Uniform,
                                 .memoryProperties = CgpuMemoryProperties::DeviceLocal | CgpuMemoryProperties::HostVisible,
                                 .size = size,
                                 .debugName = "[BumpAlloc]"
                               }, &buffer))
    {
      return nullptr;
    }

    return std::make_shared<GgpuBumpAllocator>(ctx, delayedResourceDestroyer, buffer, size);
  }

  GgpuBumpAllocator::GgpuBumpAllocator(CgpuContext* ctx,
                                       GgpuDelayedResourceDestroyer& delayedResourceDestroyer,
                                       CgpuBuffer buffer, uint32_t size)
    : m_delayedResourceDestroyer(delayedResourceDestroyer)
    , m_buffer(buffer)
    , m_size(size)
  {
    const CgpuDeviceProperties& properties = cgpuGetDeviceProperties(ctx);

    m_cpuPtr = (uint8_t*) cgpuGetBufferCpuPtr(ctx, buffer);
    m_align = properties.minUniformBufferOffsetAlignment;
  }

  GgpuBumpAllocator::~GgpuBumpAllocator()
  {
    m_delayedResourceDestroyer.enqueueDestruction(m_buffer);
  }

  CgpuBuffer GgpuBumpAllocator::getBuffer() const
  {
    return m_buffer;
  }

  // TODO: proper error handling and possibly overflow tracking
  GgpuTempAllocation<uint8_t> GgpuBumpAllocator::alloc(uint32_t size)
  {
    assert(size < m_size);

    m_offset = (m_offset + m_align - 1) / m_align * m_align;

    if (m_offset + size > m_size)
    {
      m_offset = 0;
    }

    GgpuTempAllocation<uint8_t> tmp {
      .cpuPtr = m_cpuPtr + m_offset,
      .bufferOffset = m_offset
    };

    m_offset += size;
    return tmp;
  }
}
