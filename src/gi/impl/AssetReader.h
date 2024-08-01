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
