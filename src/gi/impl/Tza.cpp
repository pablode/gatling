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

namespace gtl
{
  GiTensorDescriptions giTzaParseTensors(const uint8_t* data, size_t size)
  {
    // Header:
    //  u16 | magic
    //   u8 | major version
    //   u8 | minor version
    //  u64 | table offset
    //  u32 | number of tensors


    // For each tensor:
    //  u16 | name length
    //  u8* | name
    //   u8 | number of dimensions
    //  u32 | shape
    //  u8* | layout string
    //   u8 | data type
    //  u64 | data offset
    //  u8* | data

    return {};
  }
}
