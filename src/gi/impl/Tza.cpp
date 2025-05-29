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

#include "Tza.h"

#include <gtl/gb/Log.h>

namespace gtl
{
  static void _giTzaCheckBounds(const uint8_t* ptr, const uint8_t* end, uint64_t size)
  {
    if (end - ptr < (ptrdiff_t) size)
    {
      GB_FATAL("TZA: stream broken; expected byte");
    }
  }

  template<typename T>
  static T _giTzaRead(const uint8_t*& ptr, const uint8_t* end)
  {
    _giTzaCheckBounds(ptr, end, sizeof(T));
    T value;
    memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
  }

  GiTzaTensorDescriptions giTzaParseTensors(const uint8_t* data, size_t size)
  {
    const uint8_t* ptr = &data[0];
    const uint8_t* end = &data[size];

    // Header:
    //  u16 | magic
    //   u8 | major version
    //   u8 | minor version
    //  u64 | table offset
    //  u32 | number of tensors
    auto magic = _giTzaRead<uint16_t>(ptr, end);
    if (magic != 0x41D7)
    {
      GB_FATAL("corrupt header");
    }

    auto versionMajor = _giTzaRead<uint8_t>(ptr, end);
    auto versionMinor = _giTzaRead<uint8_t>(ptr, end);
    if (versionMajor != 2)
    {
      GB_FATAL("version mismatch");
    }

    auto tableOffset = _giTzaRead<uint64_t>(ptr, end);
    ptr = &data[tableOffset];

    auto tensorCount = _giTzaRead<uint32_t>(ptr, end);

    // For each tensor:
    //  u16 | name length
    //  u8* | name
    //   u8 | number of dimensions
    //  u32 | shape
    //  u8* | layout string
    //   u8 | data type char
    //  u64 | data offset
    //  u8* | data
    GiTzaTensorDescriptions descs;

    GB_LOG("parsing {} tensors:", tensorCount);

    for (uint32_t i = 0; i < tensorCount; i++)
    {
      auto nameLength = _giTzaRead<uint16_t>(ptr, end);
      _giTzaCheckBounds(ptr, end, nameLength);

      std::string name(ptr, ptr + nameLength);
      ptr += nameLength;

      auto dimCount = _giTzaRead<uint8_t>(ptr, end);
      uint64_t dimSize = dimCount * sizeof(int);

      std::vector<int> dimensions(dimCount);
      _giTzaCheckBounds(ptr, end, dimSize);
      memcpy(&dimensions[0], ptr, dimSize);
      ptr += dimSize;

      _giTzaCheckBounds(ptr, end, dimCount);
      std::string layoutStr(ptr, ptr + dimCount);
      ptr += dimCount;

      GiTzaTensorLayout layout;
      if (layoutStr == "x")
      {
        layout = GiTzaTensorLayout::x;
      }
      else if (layoutStr == "oihw")
      {
        layout = GiTzaTensorLayout::oihw;
      }
      else
      {
        GB_FATAL("unsupported tensor layout");
      }

      auto dataTypeChar = _giTzaRead<char>(ptr, end);
      if (dataTypeChar != 'h')
      {
        GB_FATAL("unsupported tensor data type");
      }

      auto dataOffset = _giTzaRead<uint64_t>(ptr, end);

      uint64_t dataSize = 1;
      for (uint32_t i = 0; i < dimCount; i++)
      {
        dataSize *= dimensions[i];
      }
      dataSize *= sizeof(float) / 2; // only support half

      _giTzaCheckBounds(&data[dataOffset], end, dataSize);

      GB_LOG(" {} ({}, {}, {})", name, dimCount, layoutStr, dataTypeChar);

      descs[name] = GiTzaTensorDescription{
        .dimensions = dimensions,
        .layout = layout,
        .dataOffset = dataOffset,
        .dataSize = dataSize
      };
    }

    return descs;
  }
}
