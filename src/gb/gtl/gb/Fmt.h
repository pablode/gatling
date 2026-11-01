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

#include <quill/bundled/fmt/format.h>
#include <quill/bundled/fmt/printf.h>

#define GB_FMT(fmt, ...) fmtquill::format(fmt, __VA_ARGS__)
#define GB_FMT_SPRINTF(fmt, ...) fmtquill::sprintf(fmt, __VA_ARGS__)

namespace gtl
{
  inline std::string gbFmtByteSize(uint64_t bytes)
  {
    constexpr uint64_t MIB = 1024 * 1024;
    constexpr uint64_t GIB = 1024 * MIB;

    if (bytes >= 1000 * MIB)
    {
      return GB_FMT("{:.2f} GiB", double(bytes) / double(GIB));
    }

    return GB_FMT("{:.2f} MiB", double(bytes) / double(MIB));
  }
}
