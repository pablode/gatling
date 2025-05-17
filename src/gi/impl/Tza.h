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

#include <string>
#include <vector>
#include <unordered_map>

namespace gtl
{
  enum class GiTzaTensorLayout { x, oihw };

  enum class GiTzaTensorDataType { Float16, Float32 };

  struct GiTzaTensorDescription
  {
    std::vector<int> dimensions; // CHW or OIHW
    GiTzaTensorLayout layout;
    GiTzaTensorDataType dataType;
    uint64_t dataOffset;
    uint64_t dataSize;
  };

  using GiTensorDescriptions = std::unordered_map<std::string, GiTzaTensorDescription>;

  GiTensorDescriptions giTzaParseTensors(const uint8_t* data, size_t size);
}

