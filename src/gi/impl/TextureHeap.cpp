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

#include "TextureHeap.h"

#include <gtl/gb/Log.h>
#include <gtl/ggpu/Stager.h>

namespace gtl
{
  namespace rp = shader_interface::rp_main;

  GiTextureHeap::GiTextureHeap(CgpuContext* ctx, GgpuStager& stager)
    : m_ctx(ctx)
    , m_stager(stager)
    , m_textureAllocator(rp::MAX_TEXTURE_COUNT)
    , m_textureInfoAllocator(rp::MAX_TEXTURE_COUNT)
  {
  }

  bool GiTextureHeap::initialize()
  {
    uint64_t bufferSize = uint64_t(rp::MAX_TEXTURE_COUNT) * sizeof(rp::TextureInfo);

    if (!cgpuCreateBuffer(m_ctx, {
                            .usage = CgpuBufferUsage::Storage | CgpuBufferUsage::TransferDst,
                            .memoryProperties = CgpuMemoryProperties::DeviceLocal,
                            .size = bufferSize,
                            .debugName = "TextureInfos"
                          }, &m_textureInfoBuffer))
    {
      GB_ERROR("failed to create texture info buffer");
      return false;
    }

    std::vector<rp::TextureInfo> zeroedTextureInfos(rp::MAX_TEXTURE_COUNT);
    if (!m_stager.stageToBuffer((const uint8_t*) zeroedTextureInfos.data(), bufferSize, m_textureInfoBuffer) ||
        !m_stager.flush())
    {
      GB_ERROR("failed to initialize texture info buffer");
      destroy();
      return false;
    }

    return true;
  }

  void GiTextureHeap::destroy()
  {
    if (m_textureInfoBuffer.handle)
    {
      cgpuDestroyBuffer(m_ctx, m_textureInfoBuffer);
      m_textureInfoBuffer = {};
    }
  }

  GiTextureHeapAllocation GiTextureHeap::allocateTextures(uint32_t count)
  {
    GiTextureHeapAllocation allocation = m_textureAllocator.allocate(count);
    GB_EXPECT(allocation.offset != GiTextureHeapAllocation::NO_SPACE, "max number of textures exceeded");
    return allocation;
  }

  void GiTextureHeap::freeTextures(GiTextureHeapAllocation allocation)
  {
    m_textureAllocator.free(allocation);
  }

  GiTextureHeapAllocation GiTextureHeap::allocateTextureInfos(uint32_t count)
  {
    GiTextureHeapAllocation allocation = m_textureInfoAllocator.allocate(count);
    GB_EXPECT(allocation.offset != GiTextureHeapAllocation::NO_SPACE, "max number of texture infos exceeded");
    return allocation;
  }

  void GiTextureHeap::freeTextureInfos(GiTextureHeapAllocation allocation)
  {
    m_textureInfoAllocator.free(allocation);
  }

  bool GiTextureHeap::updateTextureInfos(GiTextureHeapAllocation allocation,
                                         const std::vector<rp::TextureInfo>& textureInfos)
  {
    if (textureInfos.empty())
    {
      return true;
    }

    uint64_t offset = uint64_t(allocation.offset) * sizeof(rp::TextureInfo);
    uint64_t size = uint64_t(textureInfos.size()) * sizeof(rp::TextureInfo);

    return m_stager.stageToBuffer((const uint8_t*) textureInfos.data(), size, m_textureInfoBuffer, offset) &&
           m_stager.flush();
  }

  CgpuBuffer GiTextureHeap::textureInfoBuffer() const
  {
    return m_textureInfoBuffer;
  }
}
