//
// Copyright (C) 2025 Pablo Delgado Krämer
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

#include <utility>

namespace gtl
{
  struct GbHash
  {
    uint64_t val = 0;

    inline bool operator==(const GbHash& other)
    {
      return val == other.val;
    }
  };

  template <class T>
  inline GbHash GbHashAppend(GbHash other, const T& v)
  {
    uint64_t seed = other.val;
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return GbHash{ seed };
  }
}
