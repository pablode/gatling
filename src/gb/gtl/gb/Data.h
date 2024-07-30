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

#include <initializer_list>
#include <vector>

namespace gtl
{
// TODO: rename data -> ptr or basePtr so that
//       we don't have data.data pattern
  template<typename T, typename I = uint64_t>
  struct GbSpan
  {
    T* data;
    I size;

    GbSpan(T* _data, I _size)
      : data(_data)
      , size(_size)
    {
    }
    GbSpan(T* _data)
      : GbSpan(_data, 1)
    {
    }
    GbSpan(T* begin, T* end)
      : GbSpan(begin, (end - begin) / sizeof(T))
    {
    }
    GbSpan(std::initializer_list<T> list)
      : GbSpan(list.begin(), list.end())
    {
    }
    GbSpan(std::vector<T> v)
      : GbSpan(v.data(), v.size())
    {
    }

    T& begin()
    {
      return data[0];
    }
    T& end()
    {
      return data[size - 1];
    }
    GbSpan subspan(I pos, I len)
    {
      assert((pos + len) < size);
      return GbSpan(&data[pos], &data[pos + len]);
    }
  };

  template<typename T>
  using GbShortSpan = GbSpan<T, uint32_t>;

  template<std::integral T>
  T gbAlignUpwards(T value, T alignment)
  {
    if (alignment == T(0))
    {
      return value;
    }

    return (value + alignment - T(1)) / alignment * alignment;
  }
}
