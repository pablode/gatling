/*
 * This file is part of gatling.
 *
 * Copyright (C) 2019-2022 Pablo Delgado Krämer
 *
 * gatling is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef IMGIO_MMAP_H
#define IMGIO_MMAP_H

#include <stdbool.h>
#include <stddef.h>

struct imgio_file;

enum ImgioFileUsage
{
  IMGIO_FILE_USAGE_READ  = 1,
  IMGIO_FILE_USAGE_WRITE = 2
};

bool imgio_file_create(
  const char* path,
  size_t size,
  struct imgio_file** file
);

bool imgio_file_open(
  const char* path,
  enum ImgioFileUsage usage,
  struct imgio_file** file
);

size_t imgio_file_size(struct imgio_file* file);

bool imgio_file_close(struct imgio_file* file);

void* imgio_mmap(
  struct imgio_file* file,
  size_t offset,
  size_t size
);

bool imgio_munmap(
  struct imgio_file* file,
  void* addr
);

#endif
