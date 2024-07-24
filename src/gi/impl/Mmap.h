//
// Copyright (C) 2019 Pablo Delgado Krämer
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

#include <stdbool.h>
#include <stddef.h>

namespace gtl
{
  struct GiFile;

  enum class GiFileUsage { Read, Write };

  bool giFileCreate(const char* path, size_t size, GiFile** file);

  bool giFileOpen(const char* path, GiFileUsage usage, GiFile** file);

  size_t giFileSize(GiFile* file);

  bool giFileClose(GiFile* file);

  void* giMmap(GiFile* file, size_t offset, size_t size);

  bool giMunmap(GiFile* file, void* addr);
}
