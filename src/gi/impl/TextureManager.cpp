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
#include <gtl/ggpu/DelayedResourceDestroyer.h>
#include <gtl/ggpu/Stager.h>
#include <gtl/imgio/Imgio.h>

#include <xxhash.h>

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

  CgpuImageFormat _TranslateImageFormat(ImgioFormat format)
  {
    switch (format)
    {
    case ImgioFormat::RGBA8_UNORM: return CgpuImageFormat::R8G8B8A8Unorm;
    case ImgioFormat::R32_FLOAT: return CgpuImageFormat::R32Float;
    case ImgioFormat::BC1_UNORM: return CgpuImageFormat::Bc1Unorm;
    case ImgioFormat::BC1_UNORM_SRGB: return CgpuImageFormat::Bc1UnormSrgb;
    case ImgioFormat::BC2_UNORM: return CgpuImageFormat::Bc2Unorm;
    case ImgioFormat::BC2_UNORM_SRGB: return CgpuImageFormat::Bc2UnormSrgb;
    case ImgioFormat::BC3_UNORM: return CgpuImageFormat::Bc3Unorm;
    case ImgioFormat::BC3_UNORM_SRGB: return CgpuImageFormat::Bc3UnormSrgb;
    case ImgioFormat::BC4_UNORM: return CgpuImageFormat::Bc4Unorm;
    case ImgioFormat::BC4_SNORM: return CgpuImageFormat::Bc4Snorm;
    case ImgioFormat::BC5_UNORM: return CgpuImageFormat::Bc5Unorm;
    case ImgioFormat::BC5_SNORM: return CgpuImageFormat::Bc5Snorm;
    case ImgioFormat::BC7_UNORM: return CgpuImageFormat::Bc7Unorm;
    case ImgioFormat::BC7_UNORM_SRGB: return CgpuImageFormat::Bc7UnormSrgb;
    default: CgpuImageFormat::Undefined;
    }
  }
}

namespace gtl
{
  GiTextureManager::GiTextureManager(CgpuDevice device, GiAssetReader& assetReader, GgpuStager& stager,
                                     GgpuDelayedResourceDestroyer& delayedResourceDestroyer)
    : m_device(device)
    , m_assetReader(assetReader)
    , m_stager(stager)
    , m_delayedResourceDestroyer(delayedResourceDestroyer)
  {
  }

  void GiTextureManager::destroy()
  {
    for (const auto& pathImagePair : m_fileCache)
    {
      if (GiImagePtr image = pathImagePair.second.lock(); image)
      {
        cgpuDestroyImage(m_device, *image);
      }
    }
    for (const auto& pathImagePair : m_binaryCache)
    {
      if (GiImagePtr image = pathImagePair.second.lock(); image)
      {
        cgpuDestroyImage(m_device, *image);
      }
    }
    m_fileCache.clear();
    m_binaryCache.clear();
  }

  GiImagePtr GiTextureManager::loadTextureFromFilePath(const char* filePath, bool is3dImage, bool destroyImmediately)
  {
    auto cacheResult = m_fileCache.find(filePath);

    if (cacheResult != m_fileCache.end())
    {
      GiImagePtr image = cacheResult->second.lock();

      if (image)
      {
        GB_DEBUG("found image \"{}\" in cache", filePath);
        return image;
      }
    }

    ImgioImage imageData;
    if (!_ReadImage(filePath, m_assetReader, &imageData) || imageData.format == ImgioFormat::UNSUPPORTED)
    {
      GB_ERROR("failed to read image {}", filePath);
      return nullptr;
    }

    GB_LOG("read image \"{}\" ({:.2f} MiB)", filePath, imageData.size * BYTES_TO_MIB);

    CgpuImageCreateInfo createInfo = {
      .width = imageData.width,
      .height = imageData.height,
      .is3d = is3dImage,
      .format = _TranslateImageFormat(imageData.format),
      .debugName = filePath
    };

    GiImagePtr image = makeImagePtr(destroyImmediately);

    bool creationSuccessful = cgpuCreateImage(m_device, createInfo, image.get()) &&
                              m_stager.stageToImage(&imageData.data[0], imageData.size, *image, imageData.width, imageData.height, 1);

    if (!creationSuccessful)
    {
      GB_ERROR("failed to upload image {}", filePath);
      return nullptr;
    }

    m_fileCache[filePath] = std::weak_ptr<CgpuImage>(image);

    return image;
  }

  GiImagePtr GiTextureManager::makeImagePtr(bool destroyImmediately)
  {
    return std::shared_ptr<CgpuImage>(new CgpuImage, [=](CgpuImage* d) {
      if (destroyImmediately)
      {
        cgpuDestroyImage(m_device, *d);
      }
      else
      {
        m_delayedResourceDestroyer.enqueueDestruction(*d);
      }
      delete d;
    });
  }

  bool GiTextureManager::loadTextureDescriptions(const std::vector<McTextureDescription>& textureDescriptions,
                                                 std::vector<GiImagePtr>& images)
  {
    size_t texCount = textureDescriptions.size();

    if (texCount == 0)
    {
      return true;
    }

    GB_LOG("staging {} images", texCount);

    images.reserve(texCount);

    for (size_t i = 0; i < texCount; i++)
    {
      fflush(stdout);

      auto& textureResource = textureDescriptions[i];
      auto& payload = textureResource.data;

      CgpuImageCreateInfo createInfo;
      createInfo.is3d = textureResource.is3dImage;
      createInfo.format = textureResource.isFloat ? CgpuImageFormat::R32Float : CgpuImageFormat::R8G8B8A8Unorm;
      createInfo.usage = CgpuImageUsage::Sampled | CgpuImageUsage::TransferDst;

      const char* filePath = textureResource.filePath.c_str();
      if (strcmp(filePath, "") == 0)
      {
        uint64_t payloadSize = payload.size();
        if (payloadSize == 0)
        {
          GB_ERROR("image {} has no payload", i);
          continue;
        }

        uint64_t hash = XXH64(payload.data(), payloadSize, 0);
        if (auto it = m_binaryCache.find(hash); it != m_binaryCache.end())
        {
          GiImagePtr image = it->second.lock();

          if (image)
          {
            GB_DEBUG("found binary image {:x} in cache", hash);

            images.push_back(image);
            continue;
          }
        }

        GiImagePtr image = makeImagePtr();

        GB_LOG("image {} has binary payload of {:.2f} MiB", i, payloadSize * BYTES_TO_MIB);

        createInfo.width = textureResource.width;
        createInfo.height = textureResource.height;
        createInfo.depth = textureResource.depth;
        createInfo.debugName = "[Payload texture]";

        if (!cgpuCreateImage(m_device, createInfo, image.get()))
        {
          return false;
        }

        if (!m_stager.stageToImage(payload.data(), payloadSize, *image, createInfo.width, createInfo.height, createInfo.depth))
        {
          return false;
        }

        m_binaryCache[hash] = std::weak_ptr<CgpuImage>(image);

        images.push_back(image);
        continue;
      }

      GiImagePtr image = loadTextureFromFilePath(filePath, textureResource.is3dImage);

      if (image)
      {
        images.push_back(image);
        continue;
      }

      image = makeImagePtr();

      // use 1x1 black fallback image
      createInfo.width = 1;
      createInfo.height = 1;
      createInfo.depth = 1;

      if (!cgpuCreateImage(m_device, createInfo, image.get()))
      {
        return false;
      }

      uint8_t black[4] = { 0, 0, 0, 0 };
      if (!m_stager.stageToImage(black, 4, *image, 1, 1, 1))
      {
        return false;
      }

      images.push_back(image);
    }

    m_stager.flush();

    return true;
  }
}
