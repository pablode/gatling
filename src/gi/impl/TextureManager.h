//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
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

#include <unordered_map>
#include <string>
#include <vector>

#include <gtl/cgpu/Cgpu.h>
#include <gtl/mc/Backend.h>

namespace gtl
{
  class GgpuStager;
  class GiAssetReader;

  class GiTextureManager
  {
  public:
    GiTextureManager(CgpuDevice device, GiAssetReader& assetReader, gtl::GgpuStager& stager);

    ~GiTextureManager();

    void destroy();

  public:
    bool loadTextureFromFilePath(const char* filePath,
                                 CgpuImage& image,
                                 bool is3dImage = false,
                                 bool flushImmediately = true);

    bool loadTextureDescriptions(const std::vector<gtl::McTextureDescription>& textureDescriptions,
                                 std::vector<CgpuImage>& images);

    void evictAndDestroyCachedImage(CgpuImage image);

  private:
    CgpuDevice m_device;
    GiAssetReader& m_assetReader;
    gtl::GgpuStager& m_stager;
    // FIXME: implement a proper CPU and GPU-aware cache with eviction strategy
    std::unordered_map<std::string, CgpuImage> m_imageCache;
  };
}
