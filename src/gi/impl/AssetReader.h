#pragma once

#include <stddef.h>
#include <vector>

#include "Gi.h"

namespace gtl
{
  class GiMmapAssetReader : public GiAssetReader
  {
  public:
    GiAsset* open(const char* path) override;

    size_t size(const GiAsset* asset) const override;

    void* data(const GiAsset* asset) const override;

    void close(GiAsset* asset) override;
  };

  class GiAggregateAssetReader : public GiAssetReader
  {
  public:
    void addAssetReader(GiAssetReader* reader);

  public:
    GiAsset* open(const char* path) override;

    size_t size(const GiAsset* asset) const override;

    void* data(const GiAsset* asset) const override;

    void close(GiAsset* asset) override;

  private:
    std::vector<GiAssetReader*> m_readers;
  };
}
