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

#include <cgpu.h>

class GiAssetReader;

namespace gi
{
  namespace sg
  {
    struct TextureResource;
  }

  class Stager;

  class TexSys
  {
  public:
    TexSys(CgpuDevice device, GiAssetReader& assetReader, Stager& stager);

    ~TexSys();

    void destroy();

  public:
    bool loadTextures(const std::vector<sg::TextureResource>& textureResources,
                      std::vector<CgpuImage>& images2d,
                      std::vector<CgpuImage>& images3d);

    void destroyUncachedImages(const std::vector<CgpuImage>& images);

  private:
    CgpuDevice m_device;
    GiAssetReader& m_assetReader;
    Stager& m_stager;
    // FIXME: implement a proper CPU and GPU-aware cache with eviction strategy
    std::unordered_map<std::string, CgpuImage> m_imageCache;
  };
}
