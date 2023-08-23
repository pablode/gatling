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

#include "texsys.h"

#include "mmap.h"
#include "gi.h"

#include <sg/ShaderGen.h>
#include <stager.h>
#include <imgio.h>

#include <assert.h>
#include <string.h>
#include <inttypes.h>

using namespace gtl;

const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

namespace detail
{
  bool readImage(const char* filePath, GiAssetReader& assetReader, imgio_img* img)
  {
    GiAsset* asset = assetReader.open(filePath);
    if (!asset)
    {
      return false;
    }

    size_t size = assetReader.size(asset);
    void* data = assetReader.data(asset);

    bool loadResult = data && imgio_load_img(data, size, img) == IMGIO_OK;

    assetReader.close(asset);
    return loadResult;
  }
}

namespace gi
{
  TexSys::TexSys(CgpuDevice device, GiAssetReader& assetReader, GgpuStager& stager)
    : m_device(device)
    , m_assetReader(assetReader)
    , m_stager(stager)
  {
  }

  TexSys::~TexSys()
  {
    assert(m_imageCache.empty());
  }

  void TexSys::destroy()
  {
    for (const auto& pathImagePair : m_imageCache)
    {
      cgpuDestroyImage(m_device, pathImagePair.second);
    }
    m_imageCache.clear();
  }

  bool TexSys::loadTextureFromFilePath(const char* filePath, CgpuImage& image, bool is3dImage, bool flushImmediately)
  {
    auto cacheResult = m_imageCache.find(filePath);
    if (cacheResult != m_imageCache.end())
    {
      image = cacheResult->second;
      return true;
    }

    imgio_img imageData;
    if (!detail::readImage(filePath, m_assetReader, &imageData))
    {
      return false;
    }

    printf("image read from path %s of size %.2fMiB\n",
      filePath, imageData.size * BYTES_TO_MIB);

    CgpuImageDesc imageDesc = {
      .width = imageData.width,
      .height = imageData.height,
      .is3d = is3dImage
    };
    bool creationSuccessful = cgpuCreateImage(m_device, &imageDesc, &image) &&
                              m_stager.stageToImage(imageData.data, imageData.size, image, imageData.width, imageData.height, 1);

    imgio_free_img(&imageData);

    if (!creationSuccessful)
    {
      return false;
    }

    m_imageCache[filePath] = image;

    if (flushImmediately)
    {
      m_stager.flush();
    }

    return true;
  }

  bool TexSys::loadTextureResources(const std::vector<sg::TextureResource>& textureResources,
                                    std::vector<CgpuImage>& images2d,
                                    std::vector<CgpuImage>& images3d)
  {
    size_t texCount = textureResources.size();

    if (texCount == 0)
    {
      return true;
    }

    printf("staging %zu images\n", texCount);

    images2d.reserve(texCount);
    images3d.reserve(texCount);

    bool result;

    for (int i = 0; i < texCount; i++)
    {
      fflush(stdout);
      CgpuImage image;

      auto& textureResource = textureResources[i];
      auto& payload = textureResource.data;

      CgpuImageDesc imageDesc;
      imageDesc.is3d = textureResource.is3dImage;
      imageDesc.format = CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM;
      imageDesc.usage = CGPU_IMAGE_USAGE_FLAG_SAMPLED | CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST;

      auto& imageVector = imageDesc.is3d ? images3d : images2d;

      int binding = textureResource.binding;

      const char* filePath = textureResource.filePath.c_str();
      if (strcmp(filePath, "") == 0)
      {
        uint64_t payloadSize = payload.size();
        if (payloadSize == 0)
        {
          fprintf(stderr, "image %d has no payload\n", i);
          continue;
        }

        printf("image %d has binary payload of %.2fMiB\n", i, payloadSize * BYTES_TO_MIB);

        imageDesc.width = textureResource.width;
        imageDesc.height = textureResource.height;
        imageDesc.depth = textureResource.depth;

        if (!cgpuCreateImage(m_device, &imageDesc, &image))
          return false;

        result = m_stager.stageToImage(payload.data(), payloadSize, image, imageDesc.width, imageDesc.height, imageDesc.depth);
        if (!result) return false;

        imageVector.push_back(image);
        continue;
      }

      if (loadTextureFromFilePath(filePath, image, textureResource.is3dImage, false))
      {
        imageVector.push_back(image);
        continue;
      }

      fprintf(stderr, "failed to read image %d from path %s\n", i, filePath);
      imageDesc.width = 1;
      imageDesc.height = 1;
      imageDesc.depth = 1;

      if (!cgpuCreateImage(m_device, &imageDesc, &image))
        return false;

      uint8_t black[4] = { 0, 0, 0, 0 };
      result = m_stager.stageToImage(black, 4, image, 1, 1, 1);
      if (!result) return false;

      imageVector.push_back(image);
    }

    m_stager.flush();

    return true;
  }

  void TexSys::destroyUncachedImages(const std::vector<CgpuImage>& images)
  {
    for (CgpuImage image : images)
    {
      bool isCached = false;

      for (const auto& pathImagePair : m_imageCache)
      {
        CgpuImage cachedImage = pathImagePair.second;

        if (cachedImage.handle == image.handle)
        {
          isCached = true;
          break;
        }
      }

      if (!isCached)
      {
        cgpuDestroyImage(m_device, image);
      }
    }
  }

  void TexSys::evictAndDestroyCachedImage(CgpuImage image)
  {
    for (auto it = m_imageCache.begin(); it != m_imageCache.end(); it++)
    {
      if (it->second.handle != image.handle)
      {
        continue;
      }

      m_imageCache.erase(it);

      cgpuDestroyImage(m_device, image);

      return;
    }

    assert(false);
  }
}
