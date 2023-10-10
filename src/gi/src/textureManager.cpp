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

#include "textureManager.h"

#include "mmap.h"
#include "gi.h"

#include <GlslShaderGen.h> // TODO: remove this dependency
#include <Backend.h>
#include <stager.h>
#include <imgio.h>

#include <stack>
#include <assert.h>
#include <string.h>
#include <inttypes.h>

using namespace gtl;

const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

namespace
{
  bool _ReadImage(const char* filePath, ::GiAssetReader& assetReader, imgio_img* img)
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
    assert(m_imageCache.empty());
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

    imgio_img imageData;
    if (!_ReadImage(filePath, m_assetReader, &imageData))
    {
      return false;
    }

    printf("image read from path %s of size %.2fMiB\n",
      filePath, imageData.size * BYTES_TO_MIB);

    CgpuImageCreateInfo createInfo = {
      .width = imageData.width,
      .height = imageData.height,
      .is3d = is3dImage,
      .debugName = filePath
    };
    bool creationSuccessful = cgpuCreateImage(m_device, &createInfo, &image) &&
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

  bool GiTextureManager::loadTextureDescriptions(const std::vector<McTextureDescription>& textureDescriptions,
                                                 std::vector<CgpuImage>& images2d,
                                                 std::vector<CgpuImage>& images3d)
  {
    size_t texCount = textureDescriptions.size();

    if (texCount == 0)
    {
      return true;
    }

    printf("staging %zu images\n", texCount);

    images2d.reserve(texCount);
    images3d.reserve(texCount);

    struct _GpuUploadInfo
    {
      imgio_img img;
      uint32_t depth;
      bool is3d;
      CgpuImageFormat format;
      CgpuImageUsageFlags usage;
      const char* debugName;
      uint32_t arrayIndex;
    };

    // Helper threads that perform IO and image decoding
    constexpr static int NUM_THREADS = 8;

    int numActiveThreads = NUM_THREADS;
    std::thread threads[NUM_THREADS];

    std::stack<_GpuUploadInfo> stack;
    std::mutex stackMutex;
    std::condition_variable condVar;

    std::atomic_uint32_t img2dIdx, img3dIdx;
    std::atomic_uint32_t workItemOffset = 0; // desc array index
    for (int i = 0; i < NUM_THREADS; i++)
    {
      threads[i] = std::thread([&]() {
        while (true)
        {
          uint32_t workIdx = workItemOffset++;
          if (workIdx >= textureDescriptions.size())
          {
            break;
          }

          const McTextureDescription& desc = textureDescriptions[workIdx];
          auto& payload = desc.data;

          // Init upload data to black 1x1 pixel
          static uint8_t BLACK_PIXEL[] = { 0, 0, 0, 0 };
          _GpuUploadInfo uploadInfo = {
            .img = {
              .data = BLACK_PIXEL,
              .size = 4,
              .width = 1,
              .height = 1,
            },
            .depth = 1,
            .is3d = false,
            .format = CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM,
            .usage = CGPU_IMAGE_USAGE_FLAG_SAMPLED | CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST,
            .debugName = "INVALID",
            .arrayIndex = (desc.is3dImage ? img3dIdx++ : img2dIdx++)
          };

          const char* filePath = desc.filePath.c_str();
          bool hasPayload = strcmp(filePath, "") == 0;

          if (hasPayload)
          {
            uint64_t payloadSize = payload.size();

            if (payloadSize == 0)
            {
              fprintf(stderr, "image %d has no payload\n", workIdx);
            }
            else
            {
              printf("image %d has binary payload of %.2fMiB\n", workIdx, payloadSize * BYTES_TO_MIB);

              uploadInfo.img = {
                .data = (uint8_t*) payload.data(), // TODO: remove const cast
                .size = payloadSize,
                .width = desc.width,
                .height = desc.height
              };
              uploadInfo.depth = desc.depth;
              uploadInfo.is3d = desc.is3dImage;
            }
          }
          else
          {
            // TODO: implement cache

            if (_ReadImage(filePath, m_assetReader, &uploadInfo.img))
            {
              printf("image %i read from path %s of size %.2fMiB\n", workIdx, filePath, uploadInfo.img.size * BYTES_TO_MIB);

              uploadInfo.debugName = filePath;
            }
          }

          {
            std::lock_guard lock(stackMutex);
            stack.push(uploadInfo);
          }

          condVar.notify_one();
        }

        numActiveThreads--;
        condVar.notify_one();
      });
    }

    // Main thread uploads image data to GPU
    std::mutex condVarMutex;
    bool errorOccured = false;
    while (numActiveThreads > 0 || !stack.empty()) // TODO: stack access must be atomic... -> while-loop
    {
      {
        std::lock_guard lock(stackMutex);

        while (!stack.empty())
        {
          _GpuUploadInfo uploadInfo = stack.top();
          stack.pop();

          imgio_img& img = uploadInfo.img;

          CgpuImageCreateInfo createInfo = {
            .width = img.width,
            .height = img.height,
            .is3d = uploadInfo.is3d,
            .depth = uploadInfo.depth,
            .format = uploadInfo.format,
            .usage = uploadInfo.usage
          };

          CgpuImage imgGpu;
          errorOccured |= !cgpuCreateImage(m_device, &createInfo, &imgGpu);

          errorOccured |= !m_stager.stageToImage(img.data, img.size, imgGpu, img.width, img.height, 1);

          imgio_free_img(&img);

          auto& arr = (uploadInfo.is3d ? images3d : images2d);
          arr.resize(std::max(uint32_t(arr.size()), uploadInfo.arrayIndex + 1));
          arr[uploadInfo.arrayIndex] = imgGpu;
        }
      }

      if (numActiveThreads > 0)
      {
        std::unique_lock<std::mutex> lock(condVarMutex);
        condVar.wait(lock);
      }
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
      threads[i].join();
    }

    if (errorOccured)
    {
      fprintf(stderr, "TODO: error message\n");

      // TODO: dipose images? read vs. upload error?

      return false;
    }

    return true;
  }

  bool GiTextureManager::loadTextureDescriptionsOld(const std::vector<McTextureDescription>& textureDescriptions,
                                                    std::vector<CgpuImage>& images2d,
                                                    std::vector<CgpuImage>& images3d)
  {
    size_t texCount = textureDescriptions.size();

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

      auto& textureResource = textureDescriptions[i];
      auto& payload = textureResource.data;

      CgpuImageCreateInfo createInfo;
      createInfo.is3d = textureResource.is3dImage;
      createInfo.format = CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM;
      createInfo.usage = CGPU_IMAGE_USAGE_FLAG_SAMPLED | CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST;

      auto& imageVector = createInfo.is3d ? images3d : images2d;

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

        createInfo.width = textureResource.width;
        createInfo.height = textureResource.height;
        createInfo.depth = textureResource.depth;

        if (!cgpuCreateImage(m_device, &createInfo, &image))
          return false;

        result = m_stager.stageToImage(payload.data(), payloadSize, image, createInfo.width, createInfo.height, createInfo.depth);
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
      createInfo.width = 1;
      createInfo.height = 1;
      createInfo.depth = 1;

      if (!cgpuCreateImage(m_device, &createInfo, &image))
        return false;

      uint8_t black[4] = { 0, 0, 0, 0 };
      result = m_stager.stageToImage(black, 4, image, 1, 1, 1);
      if (!result) return false;

      imageVector.push_back(image);
    }

    m_stager.flush();

    return true;
  }

  void GiTextureManager::destroyUncachedImages(const std::vector<CgpuImage>& images)
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
