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
  using GbHash = uint64_t;

  inline GbHash GbHashCombine(GbHash hash, GbHash other)
  {
    hash ^= other + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }

  template <class T>
  inline GbHash GbHashAppend(GbHash hash, const T& v)
  {
    std::hash<T> hasher;
    hash ^= hasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
}
