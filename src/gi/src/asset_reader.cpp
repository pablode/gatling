#include "asset_reader.h"

#include "mmap.h"

struct GiMmapAsset
{
  gi_file* file;
  size_t size;
  void* data;
};

GiAsset* GiMmapAssetReader::open(const char* path)
{
  gi_file* file;
  if (!gi_file_open(path, GI_FILE_USAGE_READ, &file))
  {
    return nullptr;
  }

  size_t size = gi_file_size(file);
  void* data = gi_mmap(file, 0, size);
  if (!data)
  {
    gi_file_close(file);
    return nullptr;
  }

  auto iasset = new GiMmapAsset;
  iasset->file = file;
  iasset->size = size;
  iasset->data = data;
  return (GiAsset*) iasset;
}

size_t GiMmapAssetReader::size(const GiAsset* asset) const
{
  auto iasset = (GiMmapAsset*) asset;
  return iasset->size;
}

void* GiMmapAssetReader::data(const GiAsset* asset) const
{
  auto iasset = (GiMmapAsset*) asset;
  return iasset->data;
}

void GiMmapAssetReader::close(GiAsset* asset)
{
  auto iasset = (GiMmapAsset*) asset;
  gi_munmap(iasset->file, iasset->data);
  gi_file_close(iasset->file);
  delete iasset;
}
