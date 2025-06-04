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

#include "TextureManager.h"
#include "Gi.h"

#include <gtl/mc/Backend.h>
#include <gtl/gb/Log.h>
#include <gtl/ggpu/Stager.h>
#include <gtl/imgio/Imgio.h>

#include <assert.h>
#include <string.h>
#include <inttypes.h>

namespace
{
  using namespace gtl;

  constexpr static const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

  bool _ReadImage(const char* filePath, GiAssetReader& assetReader, ImgioImage* img)
  {
    GiAsset* asset = assetReader.open(filePath);
    if (!asset)
    {
      return false;
    }

    size_t size = assetReader.size(asset);
    void* data = assetReader.data(asset);

    bool loadResult = data && ImgioLoadImage(data, size, img) == ImgioError::None;

    assetReader.close(asset);
    return loadResult;
  }
}

namespace gtl
{
  GiTextureManager::GiTextureManager(CgpuDevice device, GiAssetReader& assetReader, GgpuStager& stager)
    : m_device(device)
    , m_assetReader(assetReader)
    , m_stager(stager)
  {
  }

  GiTextureManager::~GiTextureManager()
  {
    // TODO: use DelayedResourceDestroyer
    for (auto it = m_imageCache.begin(); it != m_imageCache.end(); it++)
    {
      cgpuDestroyImage(m_device, it->second);
    }
  }

  void GiTextureManager::destroy()
  {
    for (const auto& pathImagePair : m_imageCache)
    {
      cgpuDestroyImage(m_device, pathImagePair.second);
    }
    m_imageCache.clear();
  }

  bool GiTextureManager::loadTextureFromFilePath(const char* filePath, CgpuImage& image, bool is3dImage, bool flushImmediately)
  {
    auto cacheResult = m_imageCache.find(filePath);
    if (cacheResult != m_imageCache.end())
    {
      image = cacheResult->second;
      return true;
    }

    ImgioImage imageData;
    if (!_ReadImage(filePath, m_assetReader, &imageData))
    {
      return false;
    }

    GB_LOG("read image \"{}\" ({:.2f} MiB)", filePath, imageData.size * BYTES_TO_MIB);

    CgpuImageCreateInfo createInfo = {
      .width = imageData.width,
      .height = imageData.height,
      .is3d = is3dImage,
      .debugName = filePath
    };
    bool creationSuccessful = cgpuCreateImage(m_device, createInfo, &image) &&
                              m_stager.stageToImage(&imageData.data[0], imageData.size, image, imageData.width, imageData.height, 1);

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

  bool GiTextureManager::loadTextureDescriptions(const std::vector<McTextureDescription>& textureDescriptions,
                                                 std::vector<CgpuImage>& images)
  {
    size_t texCount = textureDescriptions.size();

    if (texCount == 0)
    {
      return true;
    }

    GB_LOG("staging {} images", texCount);

    images.reserve(texCount);

    bool result;

    for (size_t i = 0; i < texCount; i++)
    {
      fflush(stdout);
      CgpuImage image;

      auto& textureResource = textureDescriptions[i];
      auto& payload = textureResource.data;

      CgpuImageCreateInfo createInfo;
      createInfo.is3d = textureResource.is3dImage;
      createInfo.format = textureResource.isFloat ? CGPU_IMAGE_FORMAT_R32_SFLOAT : CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM;
      createInfo.usage = CGPU_IMAGE_USAGE_FLAG_SAMPLED | CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST;

      const char* filePath = textureResource.filePath.c_str();
      if (strcmp(filePath, "") == 0)
      {
        uint64_t payloadSize = payload.size();
        if (payloadSize == 0)
        {
          GB_ERROR("image {} has no payload", i);
          continue;
        }

        GB_LOG("image {} has binary payload of {:.2f} MiB", i, payloadSize * BYTES_TO_MIB);

        createInfo.width = textureResource.width;
        createInfo.height = textureResource.height;
        createInfo.depth = textureResource.depth;

        if (!cgpuCreateImage(m_device, createInfo, &image))
          return false;

        result = m_stager.stageToImage(payload.data(), payloadSize, image, createInfo.width, createInfo.height, createInfo.depth);
        if (!result) return false;

        images.push_back(image);
        continue;
      }

      if (loadTextureFromFilePath(filePath, image, textureResource.is3dImage, false))
      {
        images.push_back(image);
        continue;
      }

      GB_ERROR("failed to read image {} from path {}", i, filePath);
      createInfo.width = 1;
      createInfo.height = 1;
      createInfo.depth = 1;

      if (!cgpuCreateImage(m_device, createInfo, &image))
        return false;

      uint8_t black[4] = { 0, 0, 0, 0 };
      result = m_stager.stageToImage(black, 4, image, 1, 1, 1);
      if (!result) return false;

      images.push_back(image);
    }

    m_stager.flush();

    return true;
  }

  void GiTextureManager::evictAndDestroyCachedImage(CgpuImage image)
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
