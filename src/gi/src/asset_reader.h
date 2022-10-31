#pragma once

#include <stddef.h>

#include "gi.h"

class GiMmapAssetReader : public GiAssetReader
{
public:
  GiAsset* open(const char* path) override;

  size_t size(const GiAsset* asset) const override;

  void* data(const GiAsset* asset) const override;

  void close(GiAsset* asset) override;
};

