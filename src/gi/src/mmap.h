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

#pragma once

#include <stdbool.h>
#include <stddef.h>

struct gi_file;

enum GiFileUsage
{
  GI_FILE_USAGE_READ  = 1,
  GI_FILE_USAGE_WRITE = 2
};

bool gi_file_create(const char* path, size_t size, gi_file** file);

bool gi_file_open(const char* path, GiFileUsage usage, gi_file** file);

size_t gi_file_size(gi_file* file);

bool gi_file_close(gi_file* file);

void* gi_mmap(gi_file* file, size_t offset, size_t size);

bool gi_munmap(gi_file* file, void* addr);

