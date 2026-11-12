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
#include <vector>

#include <gtl/cgpu/Cgpu.h>

#include <offsetAllocator.hpp>

#include "interface/rp_main.h"

namespace gtl
{
  class GgpuStager;

  using GiTextureHeapAllocation = OffsetAllocator::Allocation;

  class GiTextureHeap
  {
  public:
    GiTextureHeap(CgpuContext* ctx, GgpuStager& stager);

    bool initialize();

    void destroy();

    GiTextureHeapAllocation allocateTextures(uint32_t count);
    void freeTextures(GiTextureHeapAllocation allocation);

    GiTextureHeapAllocation allocateTextureInfos(uint32_t count);
    void freeTextureInfos(GiTextureHeapAllocation allocation);

    bool updateTextureInfos(GiTextureHeapAllocation allocation,
                            const std::vector<shader_interface::rp_main::TextureInfo>& textureInfos);

    CgpuBuffer textureInfoBuffer() const;

  private:
    CgpuContext* m_ctx;
    GgpuStager& m_stager;
    OffsetAllocator::Allocator m_textureAllocator;
    OffsetAllocator::Allocator m_textureInfoAllocator;
    CgpuBuffer m_textureInfoBuffer = {};
  };
}
