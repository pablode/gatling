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
  class GgpuDelayedResourceDestroyer;
  class GgpuStager;
  class GiAssetReader;

  using GiImagePtr = std::shared_ptr<CgpuImage>;

  class GiTextureManager
  {
  public:
    GiTextureManager(CgpuDevice device, GiAssetReader& assetReader, gtl::GgpuStager& stager,
                     GgpuDelayedResourceDestroyer& delayedResourceDestroyer);

    void housekeep();

    void destroy();

  public:
    GiImagePtr loadTextureFromFilePath(const char* filePath,
                                       bool is3dImage = false,
                                       bool flushImmediately = true);

    bool loadTextureDescriptions(const std::vector<gtl::McTextureDescription>& textureDescriptions,
                                 std::vector<GiImagePtr>& images);

  private:
    GiImagePtr makeImagePtr();

  private:
    CgpuDevice m_device;
    GiAssetReader& m_assetReader;
    gtl::GgpuStager& m_stager;
    GgpuDelayedResourceDestroyer& m_delayedResourceDestroyer;
    std::unordered_map<std::string, std::weak_ptr<CgpuImage>> m_imageCache;
  };
}
