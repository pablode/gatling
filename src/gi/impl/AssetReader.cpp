//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#include "AssetReader.h"
#include "Mmap.h"

namespace gtl
{
  struct GiMmapAsset
  {
    GiFile* file;
    size_t size;
    void* data;
  };

  GiAsset* GiMmapAssetReader::open(const char* path)
  {
    GiFile* file;
    if (!giFileOpen(path, GiFileUsage::Read, &file))
    {
      return nullptr;
    }

    size_t size = giFileSize(file);
    void* data = giMmap(file, 0, size);
    if (!data)
    {
      giFileClose(file);
      return nullptr;
    }

    auto iasset = new GiMmapAsset;
    iasset->file = file;
    iasset->size = size;
    iasset->data = data;
    return (GiAsset*)iasset;
  }

  size_t GiMmapAssetReader::size(const GiAsset* asset) const
  {
    auto iasset = (GiMmapAsset*)asset;
    return iasset->size;
  }

  void* GiMmapAssetReader::data(const GiAsset* asset) const
  {
    auto iasset = (GiMmapAsset*)asset;
    return iasset->data;
  }

  void GiMmapAssetReader::close(GiAsset* asset)
  {
    auto iasset = (GiMmapAsset*)asset;
    giMunmap(iasset->file, iasset->data);
    giFileClose(iasset->file);
    delete iasset;
  }

  struct GiAggregateAsset
  {
    GiAssetReader* reader;
    GiAsset* asset;
  };

  void GiAggregateAssetReader::addAssetReader(GiAssetReader* reader)
  {
    m_readers.push_back(reader);
  }

  GiAsset* GiAggregateAssetReader::open(const char* path)
  {
    for (GiAssetReader* reader : m_readers)
    {
      GiAsset* asset = reader->open(path);
      if (!asset)
      {
        continue;
      }

      auto iasset = new GiAggregateAsset;
      iasset->reader = reader;
      iasset->asset = asset;
      return (GiAsset*)iasset;
    }
    return nullptr;
  }

  size_t GiAggregateAssetReader::size(const GiAsset* asset) const
  {
    auto iasset = (GiAggregateAsset*)asset;
    return iasset->reader->size(iasset->asset);
  }

  void* GiAggregateAssetReader::data(const GiAsset* asset) const
  {
    auto iasset = (GiAggregateAsset*)asset;
    return iasset->reader->data(iasset->asset);
  }

  void GiAggregateAssetReader::close(GiAsset* asset)
  {
    auto iasset = (GiAggregateAsset*)asset;
    iasset->reader->close(iasset->asset);
    delete iasset;
  }
}
