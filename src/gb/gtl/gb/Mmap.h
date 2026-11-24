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
  struct GbFile;

  enum class GbFileUsage { Read, Write };

  bool gbFileCreate(const char* path, size_t size, GbFile** file);

  bool gbFileOpen(const char* path, GbFileUsage usage, GbFile** file);

  bool gbFileExists(const char* path);

  size_t gbFileSize(GbFile* file);

  bool gbFileClose(GbFile* file);

  void* gbMmap(GbFile* file, size_t offset, size_t size);

  bool gbMunmap(GbFile* file, void* addr);
}
