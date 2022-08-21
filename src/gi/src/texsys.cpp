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

#include "stager.h"

#include <ShaderGen.h>
#include <imgio.h>
#include <assert.h>

const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

namespace gi
{
  TexSys::TexSys(cgpu_device device, Stager& stager)
    : m_device(device)
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
      cgpu_destroy_image(m_device, pathImagePair.second);
    }
    m_imageCache.clear();
  }

  bool TexSys::loadTextures(const std::vector<sg::TextureResource>& textureResources,
                            std::vector<cgpu_image>& images2d,
                            std::vector<cgpu_image>& images3d,
                            std::vector<uint16_t>& imageMappings)
  {
    uint32_t texCount = textureResources.size();

    if (texCount == 0)
    {
      return true;
    }

    printf("staging %d images\n", texCount);

    imageMappings.resize(texCount, 0);
    images2d.reserve(texCount);
    images3d.reserve(texCount);

    bool result;

    for (int i = 0; i < texCount; i++)
    {
      fflush(stdout);
      cgpu_image image = { CGPU_INVALID_HANDLE };

      auto& textureResource = textureResources[i];
      auto& payload = textureResource.data;

      cgpu_image_description image_desc;
      image_desc.is3d = textureResource.is3dImage;
      image_desc.format = CGPU_IMAGE_FORMAT_R8G8B8A8_UNORM;
      image_desc.usage = CGPU_IMAGE_USAGE_FLAG_SAMPLED | CGPU_IMAGE_USAGE_FLAG_TRANSFER_DST;

      auto& imageVector = image_desc.is3d ? images3d : images2d;

      int binding = textureResource.binding;
      assert(binding < imageMappings.size());
      imageMappings[binding] = imageVector.size();

      uint32_t payloadSize = payload.size();
      if (payloadSize > 0)
      {
        printf("image %d has binary payload of %.2fMiB\n", i, payloadSize * BYTES_TO_MIB);

        image_desc.width = textureResource.width;
        image_desc.height = textureResource.height;
        image_desc.depth = textureResource.depth;

        result = cgpu_create_image(m_device, &image_desc, &image) == CGPU_OK;
        if (!result) return false;

        result = m_stager.stageToImage(payload.data(), payloadSize, image);
        if (!result) return false;

        imageVector.push_back(image);
        continue;
      }

      const char* filePath = textureResource.filePath.c_str();

      auto cacheResult = m_imageCache.find(filePath);
      if (cacheResult != m_imageCache.end())
      {
        printf("image %d found in cache\n", i);
        imageVector.push_back(cacheResult->second);
        continue;
      }

      imgio_img image_data;
      if (imgio_load_img(filePath, &image_data) == IMGIO_OK)
      {
        printf("image %d read from path %s of size %.2fMiB\n",
          i, filePath, image_data.size * BYTES_TO_MIB);

        image_desc.width = image_data.width;
        image_desc.height = image_data.height;
        image_desc.depth = 1;

        result = cgpu_create_image(m_device, &image_desc, &image) == CGPU_OK;
        if (!result) return false;

        result = m_stager.stageToImage(image_data.data, image_data.size, image);
        if (!result) return false;

        imgio_free_img(&image_data);

        m_imageCache[filePath] = image;

        imageVector.push_back(image);
        continue;
      }

      fprintf(stderr, "failed to read image %d from path %s\n", i, filePath);
      image_desc.width = 1;
      image_desc.height = 1;
      image_desc.depth = 1;

      result = cgpu_create_image(m_device, &image_desc, &image) == CGPU_OK;
      if (!result) return false;

      uint8_t black[4] = { 0, 0, 0, 0 };
      result = m_stager.stageToImage(black, 4, image);
      if (!result) return false;

      imageVector.push_back(image);
    }

    return true;
  }

  void TexSys::destroyUncachedImages(const std::vector<cgpu_image>& images)
  {
    for (cgpu_image image : images)
    {
      bool isCached = false;

      for (const auto& pathImagePair : m_imageCache)
      {
        cgpu_image cachedImage = pathImagePair.second;

        if (cachedImage.handle == image.handle)
        {
          isCached = true;
          break;
        }
      }

      if (!isCached)
      {
        cgpu_destroy_image(m_device, image);
      }
    }
  }
}
