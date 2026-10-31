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
#include <gtl/gb/ThreadPool.h>
#include <gtl/ggpu/DeleteQueue.h>
#include <gtl/ggpu/Stager.h>
#include <gtl/imgio/Imgio.h>

#include <xxhash.h>

#include <assert.h>
#include <string.h>
#include <inttypes.h>
#include <charconv>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <unordered_set>

namespace fs = std::filesystem;

namespace
{
  using namespace gtl;

  constexpr static const float BYTES_TO_MIB = 1.0f / (1024.0f * 1024.0f);

  bool _ReadImage(const char* filePath, GiAssetReader& assetReader, ImgioImage* img, ImgioLoadFlags flags)
  {
    GiAsset* asset = assetReader.open(filePath);
    if (!asset)
    {
      return false;
    }

    size_t size = assetReader.size(asset);
    void* data = assetReader.data(asset);

    bool loadResult = data && ImgioLoadImage(data, size, img, flags) == ImgioError::None;

    assetReader.close(asset);
    return loadResult;
  }

  ImgioLoadFlags _MakeLoadFlags(bool keepHdr, bool greyscaleChannelToRgb)
  {
    ImgioLoadFlags flags = ImgioLoadFlags::None;
    if (keepHdr)
    {
      flags |= ImgioLoadFlags::KeepHdr;
    }
    if (greyscaleChannelToRgb)
    {
      flags |= ImgioLoadFlags::GreyscaleChannelToRgb;
    }
    return flags;
  }

  CgpuImageFormat _TranslateImageFormat(ImgioFormat format)
  {
    switch (format)
    {
    case ImgioFormat::RGBA8_UNORM: return CgpuImageFormat::R8G8B8A8Unorm;
    case ImgioFormat::RGB16_FLOAT: return CgpuImageFormat::R16G16B16Sfloat;
    case ImgioFormat::RGBA16_FLOAT: return CgpuImageFormat::R16G16B16A16Sfloat;
    case ImgioFormat::R32_FLOAT: return CgpuImageFormat::R32Sfloat;
    default: return CgpuImageFormat::Undefined;
    }
  }

  enum class UvTileMode
  {
    None,
    Udim,
    UvTile0,
    UvTile1
  };

  struct UvTileMask
  {
    UvTileMode mode = UvTileMode::None;
    std::string prefix;
    std::string suffix;
  };

  struct UvTileEntry
  {
    int32_t u;
    int32_t v;
    std::string path;
  };

  UvTileMask _SplitUvTileMask(const std::string& filePath)
  {
    struct Marker
    {
      const char* str;
      UvTileMode mode;
    };
    constexpr static Marker MARKERS[] = {
      { "<UDIM>", UvTileMode::Udim },
      { "<UVTILE0>", UvTileMode::UvTile0 },
      { "<UVTILE1>", UvTileMode::UvTile1 },
    };

    for (const Marker& marker : MARKERS)
    {
      size_t pos = filePath.find(marker.str);
      if (pos == std::string::npos)
      {
        continue;
      }

      return UvTileMask {
        .mode = marker.mode,
        .prefix = filePath.substr(0, pos),
        .suffix = filePath.substr(pos + strlen(marker.str))
      };
    }

    return {};
  }

  std::string _ExpandUvTileMask(const UvTileMask& mask, int32_t u, int32_t v)
  {
    std::string tile;

    switch (mask.mode)
    {
    case UvTileMode::Udim: {
      // UDIM expands to the four digit number 1000 + (u + 1) + v * 10
      uint32_t number = 1000 + uint32_t(u + 1) + uint32_t(v * 10);
      tile = std::to_string(number);
      break;
    }
    case UvTileMode::UvTile0:
      tile = "_u" + std::to_string(u) + "_v" + std::to_string(v);
      break;
    case UvTileMode::UvTile1:
      tile = "_u" + std::to_string(u + 1) + "_v" + std::to_string(v + 1);
      break;
    default:
      assert(false);
      break;
    }

    return mask.prefix + tile + mask.suffix;
  }

  bool _ParseInt(const char*& str, const char* end, int32_t& value)
  {
    auto [ptr, ec] = std::from_chars(str, end, value);
    if (ec != std::errc())
    {
      return false;
    }

    str = ptr;
    return true;
  }

  bool _ParseUvTile(const std::string& tile, UvTileMode mode, int32_t& u, int32_t& v)
  {
    if (mode == UvTileMode::Udim)
    {
      // Only 1001-1100 are valid UDIM tile numbers.
      if (tile.size() != 4)
      {
        return false;
      }

      int32_t number;
      auto [ptr, ec] = std::from_chars(tile.data(), tile.data() + tile.size(), number);
      if (ec != std::errc() || ptr != tile.data() + tile.size() || number < 1001 || number > 1100)
      {
        return false;
      }

      u = (number - 1001) % 10;
      v = (number - 1001) / 10;
      return true;
    }

    // UVTILE expands to "_u<u>_v<v>"
    const char* str = tile.data();
    const char* end = str + tile.size();

    if (end - str < 2 || str[0] != '_' || str[1] != 'u')
    {
      return false;
    }
    str += 2;

    int32_t tileU, tileV;
    if (!_ParseInt(str, end, tileU) || end - str < 2 || str[0] != '_' || str[1] != 'v')
    {
      return false;
    }
    str += 2;

    if (!_ParseInt(str, end, tileV) || str != end)
    {
      return false;
    }

    int32_t offset = (mode == UvTileMode::UvTile0) ? 0 : 1;
    if (tileU < INT8_MIN + offset || tileU > INT8_MAX + offset ||
        tileV < INT8_MIN + offset || tileV > INT8_MAX + offset)
    {
      return false;
    }

    u = tileU - offset;
    v = tileV - offset;
    return true;
  }

  bool _EnumerateUvTilesFromDirectory(const UvTileMask& mask, std::vector<UvTileEntry>& tiles)
  {
    fs::path prefixPath(mask.prefix);
    fs::path dir = prefixPath.parent_path();
    std::string filePrefix = prefixPath.filename().string();

    // The marker has to be part of the file name.
    if (mask.suffix.find('/') != std::string::npos || mask.suffix.find('\\') != std::string::npos)
    {
      return false;
    }

    if (dir.empty())
    {
      dir = ".";
    }

    std::error_code ec;
    if (!fs::is_directory(dir, ec))
    {
      return false;
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(dir, ec))
    {
      std::string fileName = entry.path().filename().string();

      if (fileName.size() <= filePrefix.size() + mask.suffix.size() ||
          !fileName.starts_with(filePrefix) || !fileName.ends_with(mask.suffix))
      {
        continue;
      }

      std::string tile = fileName.substr(filePrefix.size(), fileName.size() - filePrefix.size() - mask.suffix.size());

      int32_t u, v;
      if (!_ParseUvTile(tile, mask.mode, u, v))
      {
        continue;
      }

      tiles.push_back({ u, v, entry.path().string() });
    }

    return true;
  }

  void _EnumerateUvTilesByProbing(const UvTileMask& mask, GiAssetReader& assetReader,
                                  std::vector<UvTileEntry>& tiles)
  {
    int32_t minTile = (mask.mode == UvTileMode::Udim) ? 0 : -8;
    int32_t maxTile = (mask.mode == UvTileMode::Udim) ? 9 : 15;

    for (int32_t v = minTile; v <= maxTile; v++)
    {
      for (int32_t u = minTile; u <= maxTile; u++)
      {
        std::string path = _ExpandUvTileMask(mask, u, v);
        if (assetReader.exists(path.c_str()))
        {
          tiles.push_back({ u, v, std::move(path) });
        }
      }
    }
  }

  void _EnumerateUvTiles(const UvTileMask& mask, GiAssetReader& assetReader,
                         std::vector<UvTileEntry>& tiles)
  {
    // Packaged paths (e.g. "asset.usdz[texture.<UDIM>.exr]") cannot be enumerated through the
    // file system; resolve the individual tiles through the asset reader instead.
    bool isPackaged = mask.prefix.find('[') != std::string::npos;

    if (!isPackaged && _EnumerateUvTilesFromDirectory(mask, tiles) && !tiles.empty())
    {
      return;
    }

    tiles.clear();
    _EnumerateUvTilesByProbing(mask, assetReader, tiles);
  }
}

namespace gtl
{
  GiTextureManager::GiTextureManager(CgpuContext* ctx, GiAssetReader& assetReader, GgpuStager& stager,
                                     GgpuDeleteQueue& deleteQueue, GbThreadPool& threadPool)
    : m_ctx(ctx)
    , m_assetReader(assetReader)
    , m_stager(stager)
    , m_deleteQueue(deleteQueue)
    , m_threadPool(threadPool)
  {
    CgpuImageCreateInfo createInfo = {
      .width = 1,
      .height = 1,
      .debugName = "[Fallback texture]"
    };

    CgpuImage image = {};
    uint8_t black[4] = { 0, 0, 0, 255 };
    if (!cgpuCreateImage(m_ctx, createInfo, &image) ||
        !m_stager.stageToImage(black, 4, image, 1, 1, 1))
    {
      GB_ERROR("failed to create fallback image");
      if (image)
      {
        cgpuDestroyImage(m_ctx, image);
      }
      return;
    }

    m_fallbackImage = std::shared_ptr<CgpuImage>(new CgpuImage(image), [deleteQueue = &m_deleteQueue](CgpuImage* d) {
      deleteQueue->pushBack(*d);
      delete d;
    });
  }

  void GiTextureManager::destroy()
  {
    m_fallbackImage.reset();

    for (const auto& pathImagePair : m_fileCache)
    {
      if (GiImagePtr image = pathImagePair.second.lock(); image)
      {
        cgpuDestroyImage(m_ctx, *image);
      }
    }
    for (const auto& pathImagePair : m_binaryCache)
    {
      if (GiImagePtr image = pathImagePair.second.lock(); image)
      {
        cgpuDestroyImage(m_ctx, *image);
      }
    }
    m_fileCache.clear();
    m_binaryCache.clear();
  }

  void GiTextureManager::setGreyscaleChannelToRgb(bool enabled)
  {
    if (m_greyscaleChannelToRgb == enabled)
    {
      return;
    }

    m_greyscaleChannelToRgb = enabled;
    m_fileCache.clear(); // we need to reload because channel layout changed
  }

  GiImagePtr GiTextureManager::loadTextureFromFilePath(const char* filePath, bool destroyImmediately, bool keepHdr)
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

    ImgioLoadFlags loadFlags = _MakeLoadFlags(keepHdr, m_greyscaleChannelToRgb);

    ImgioImage imageData;
    if (!_ReadImage(filePath, m_assetReader, &imageData, loadFlags) || imageData.format == ImgioFormat::UNSUPPORTED)
    {
      return nullptr;
    }

    GB_LOG("read image \"{}\" ({:.2f} MiB)", filePath, imageData.size * BYTES_TO_MIB);

    return createImage(filePath, imageData, destroyImmediately, keepHdr);
  }

  GiImagePtr GiTextureManager::createImage(const char* filePath, const ImgioImage& imageData,
                                           bool destroyImmediately, bool keepHdr)
  {
    CgpuImageCreateInfo createInfo = {
      .width = imageData.width,
      .height = imageData.height,
      .format = _TranslateImageFormat(imageData.format),
      .debugName = filePath
    };

    GiImagePtr image = makeImagePtr(destroyImmediately);

    bool creationSuccessful = cgpuCreateImage(m_ctx, createInfo, image.get()) &&
                              m_stager.stageToImage(&imageData.data[0], imageData.size, *image, imageData.width, imageData.height, 1, keepHdr ? 8 : 4);

    if (!creationSuccessful)
    {
      GB_ERROR("failed to upload image {}", filePath);
      return nullptr;
    }

    m_fileCache[filePath] = std::weak_ptr<CgpuImage>(image);

    return image;
  }

  GiImagePtr GiTextureManager::getFallbackImage()
  {
    return m_fallbackImage;
  }

  GiImagePtr GiTextureManager::makeImagePtr(bool destroyImmediately)
  {
    return std::shared_ptr<CgpuImage>(new CgpuImage, [this, destroyImmediately](CgpuImage* d) {
      if (destroyImmediately)
      {
        cgpuDestroyImage(m_ctx, *d);
      }
      else
      {
        m_deleteQueue.pushBack(*d);
      }
      delete d;
    });
  }

  bool GiTextureManager::loadTextureDescriptions(const std::vector<McTextureDescription>& textureDescriptions,
                                                 std::vector<GiTextureResource>& resources)
  {
    size_t texCount = textureDescriptions.size();

    resources.clear();
    resources.resize(texCount);

    if (texCount == 0)
    {
      return true;
    }

    GB_LOG("loading {} texture resources", texCount);

    struct ResourceLoadInfo
    {
      bool isUvTile = false;
      int32_t uvtileMinU = 0;
      int32_t uvtileMinV = 0;
      uint32_t uvtileWidth = 0;
      uint32_t uvtileHeight = 0;
      std::vector<int32_t> imageRequests; // one entry per image slot; -1 if missing
    };

    struct LoadRequest
    {
      std::string path;
      ImgioImage imageData;
    };

    std::vector<ResourceLoadInfo> loadInfos(texCount);
    std::vector<LoadRequest> loadRequests;
    std::unordered_map<std::string, int32_t> pathToRequest;
    std::vector<uint32_t> loadIndices;
    std::unordered_set<std::string> scheduledPaths;

    auto requestImage = [&](const std::string& path) -> int32_t {
      auto it = pathToRequest.find(path);
      if (it != pathToRequest.end())
      {
        return it->second;
      }

      int32_t index = (int32_t) loadRequests.size();
      pathToRequest.emplace(path, index);
      loadRequests.push_back({ .path = path });

      auto cacheResult = m_fileCache.find(path);
      if (cacheResult != m_fileCache.end() && cacheResult->second.lock())
      {
        GB_DEBUG("found image \"{}\" in cache", path);
        return index;
      }

      if (scheduledPaths.insert(path).second)
      {
        loadIndices.push_back((uint32_t) index);
      }
      return index;
    };

    // Collect images to load and resolve uv-tile layouts.
    for (size_t i = 0; i < texCount; i++)
    {
      const McTextureDescription& textureDescription = textureDescriptions[i];

      if (textureDescription.filePath.empty())
      {
        continue; // binary payload texture
      }

      UvTileMask mask = _SplitUvTileMask(textureDescription.filePath);
      if (mask.mode == UvTileMode::None)
      {
        loadInfos[i].imageRequests.push_back(requestImage(textureDescription.filePath));
        continue;
      }

      std::vector<UvTileEntry> tiles;
      _EnumerateUvTiles(mask, m_assetReader, tiles);

      if (tiles.empty())
      {
        GB_WARN("no uv-tiles found for image \"{}\"", textureDescription.filePath);
        continue;
      }

      int32_t minU = INT32_MAX, maxU = INT32_MIN;
      int32_t minV = INT32_MAX, maxV = INT32_MIN;
      for (const UvTileEntry& tile : tiles)
      {
        minU = std::min(minU, tile.u);
        maxU = std::max(maxU, tile.u);
        minV = std::min(minV, tile.v);
        maxV = std::max(maxV, tile.v);
      }

      ResourceLoadInfo& info = loadInfos[i];
      info.isUvTile = true;
      info.uvtileMinU = minU;
      info.uvtileMinV = minV;
      info.uvtileWidth = uint32_t(maxU - minU + 1);
      info.uvtileHeight = uint32_t(maxV - minV + 1);

      // Guard against pathological tile layouts (UVTILE0/UVTILE1 indices are unbounded).
      constexpr static const uint32_t MAX_UVTILE_COUNT = 4096;
      if (uint64_t(info.uvtileWidth) * info.uvtileHeight > MAX_UVTILE_COUNT)
      {
        GB_ERROR("uv-tile set \"{}\" exceeds the maximum tile count", textureDescription.filePath);
        info = {};
        continue;
      }

      info.imageRequests.assign(size_t(info.uvtileWidth) * info.uvtileHeight, -1);

      for (const UvTileEntry& tile : tiles)
      {
        uint32_t x = uint32_t(tile.u - minU);
        uint32_t y = uint32_t(tile.v - minV);
        info.imageRequests[size_t(y) * info.uvtileWidth + x] = requestImage(tile.path);
      }
    }

    // Load images in parallel.
    const ImgioLoadFlags loadFlags = _MakeLoadFlags(false, m_greyscaleChannelToRgb);
    std::mutex readyMutex;
    std::condition_variable readyCv;
    std::deque<uint32_t> readyIndices;
    uint32_t pendingLoads = (uint32_t) loadIndices.size();

    for (uint32_t i : loadIndices)
    {
      m_threadPool.submit([&, i]() {
        LoadRequest& request = loadRequests[i];

        bool success = _ReadImage(request.path.c_str(), m_assetReader, &request.imageData, loadFlags) &&
                       request.imageData.format != ImgioFormat::UNSUPPORTED;

        if (success)
        {
          GB_LOG("read image \"{}\" ({:.2f} MiB)", request.path, request.imageData.size * BYTES_TO_MIB);
        }
        else
        {
          request.imageData = {};
        }

        {
          std::lock_guard<std::mutex> lock(readyMutex);
          if (success)
          {
            readyIndices.push_back(i);
          }
          pendingLoads--;
        }
        readyCv.notify_one();
      });
    }

    std::vector<GiImagePtr> requestImages(loadRequests.size());

    while (true)
    {
      uint32_t i;

      {
        std::unique_lock<std::mutex> lock(readyMutex);
        readyCv.wait(lock, [&]() { return !readyIndices.empty() || pendingLoads == 0; });

        if (readyIndices.empty())
        {
          break;
        }

        i = readyIndices.front();
        readyIndices.pop_front();
      }

      LoadRequest& request = loadRequests[i];
      requestImages[i] = createImage(request.path.c_str(), request.imageData, false, false);
      request.imageData = {};
    }

    // Resolve images that were already in the cache.
    for (size_t i = 0; i < loadRequests.size(); i++)
    {
      if (requestImages[i])
      {
        continue;
      }

      auto cacheResult = m_fileCache.find(loadRequests[i].path);
      if (cacheResult != m_fileCache.end())
      {
        requestImages[i] = cacheResult->second.lock();
      }
    }

    GiImagePtr fallbackImage = getFallbackImage();
    if (!fallbackImage)
    {
      return false;
    }

    for (size_t i = 0; i < texCount; i++)
    {
      const McTextureDescription& textureResource = textureDescriptions[i];
      GiTextureResource& resource = resources[i];

      if (textureResource.filePath.empty())
      {
        CgpuImageCreateInfo createInfo;
        createInfo.is3d = textureResource.is3dImage;
        createInfo.format = textureResource.isFloat ? CgpuImageFormat::R32Sfloat : CgpuImageFormat::R8G8B8A8Unorm;

        const std::vector<uint8_t>& payload = textureResource.data;
        uint64_t payloadSize = payload.size();
        if (payloadSize == 0)
        {
          GB_ERROR("image {} has no payload", i);
          resource.images.push_back(fallbackImage);
          continue;
        }

        uint64_t hash = XXH64(payload.data(), payloadSize, 0);
        if (auto it = m_binaryCache.find(hash); it != m_binaryCache.end())
        {
          GiImagePtr image = it->second.lock();

          if (image)
          {
            GB_DEBUG("found binary image {:x} in cache", hash);
            resource.images.push_back(image);
            continue;
          }
        }

        GiImagePtr image = makeImagePtr();

        GB_LOG("image {} has binary payload of {:.2f} MiB", i, payloadSize * BYTES_TO_MIB);

        createInfo.width = textureResource.width;
        createInfo.height = textureResource.height;
        createInfo.depth = textureResource.depth;
        createInfo.debugName = "[Payload texture]";

        if (!cgpuCreateImage(m_ctx, createInfo, image.get()) ||
            !m_stager.stageToImage(payload.data(), payloadSize, *image, createInfo.width, createInfo.height, createInfo.depth))
        {
          return false;
        }

        m_binaryCache[hash] = std::weak_ptr<CgpuImage>(image);

        resource.images.push_back(image);
        continue;
      }

      const ResourceLoadInfo& info = loadInfos[i];

      if (info.isUvTile)
      {
        resource.uvtileMinU = info.uvtileMinU;
        resource.uvtileMinV = info.uvtileMinV;
        resource.uvtileWidth = info.uvtileWidth;
        resource.uvtileHeight = info.uvtileHeight;

        resource.images.reserve(size_t(info.uvtileWidth) * info.uvtileHeight);

        for (int32_t requestIndex : info.imageRequests)
        {
          GiImagePtr image = (requestIndex >= 0) ? requestImages[requestIndex] : nullptr;

          if (!image)
          {
            image = fallbackImage;
          }

          resource.images.push_back(image);
        }
        continue;
      }

      GiImagePtr image;

      if (!info.imageRequests.empty())
      {
        image = requestImages[info.imageRequests[0]];
      }

      if (!image)
      {
        GB_ERROR("failed to read image {} from path {}", i, textureResource.filePath);
        image = fallbackImage;
      }

      resource.images.push_back(image);
    }

    m_stager.flush();

    return true;
  }
}
