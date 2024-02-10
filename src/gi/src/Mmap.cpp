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

#include "Mmap.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#if defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#define MAX_MAPPED_MEM_RANGES 16

struct gi_mapped_posix_range
{
  void* addr;
  size_t size;
};

struct gi_file
{
  GiFileUsage usage;
  size_t      size;
#if defined(_WIN32)
  HANDLE      file_handle;
  HANDLE      mapping_handle;
#else
  int         file_descriptor;
  gi_mapped_posix_range mapped_ranges[MAX_MAPPED_MEM_RANGES];
#endif
};

#if defined (_WIN32)

bool gi_file_create(const char* path, size_t size, gi_file** file)
{
  DWORD creation_disposition = CREATE_ALWAYS;
  DWORD desired_access = GENERIC_READ | GENERIC_WRITE;
  DWORD share_mode = FILE_SHARE_WRITE;
  DWORD flags_and_attributes = FILE_ATTRIBUTE_NORMAL;
  HANDLE file_template = NULL;
  LPSECURITY_ATTRIBUTES security_attributes = NULL;

  HANDLE file_handle = CreateFileA(
    path,
    desired_access,
    share_mode,
    security_attributes,
    creation_disposition,
    flags_and_attributes,
    file_template
  );

  if (file_handle == INVALID_HANDLE_VALUE)
  {
    return false;
  }

  DWORD protection_flags = PAGE_READWRITE;
  DWORD maximum_size_low = size & 0x00000000FFFFFFFF;
  DWORD maximum_size_high = size >> 32;

  /* "If an application specifies a size for the file mapping object that is
   * larger than the size of the actual named file on disk and if the page
   * protection allows write access, then the file on disk is increased to
   * match the specified size of the file mapping object." (MSDN, 2020-04-09) */
  HANDLE mapping_handle = CreateFileMappingA(
    file_handle,
    security_attributes,
    protection_flags,
    maximum_size_high,
    maximum_size_low,
    NULL
  );

  if (!mapping_handle)
  {
    CloseHandle(file_handle);
    return false;
  }

  (*file) = new gi_file;
  (*file)->usage = GI_FILE_USAGE_WRITE;
  (*file)->file_handle = file_handle;
  (*file)->mapping_handle = mapping_handle;
  (*file)->size = size;

  return true;
}

bool gi_file_open(const char* path, GiFileUsage usage, gi_file** file)
{
  DWORD desired_access;
  DWORD share_mode;
  DWORD protection_flags;
  if (usage == GI_FILE_USAGE_READ)
  {
    desired_access = GENERIC_READ;
    share_mode = FILE_SHARE_READ;
    protection_flags = PAGE_READONLY;
  }
  else if (usage == GI_FILE_USAGE_WRITE)
  {
    desired_access = GENERIC_READ | GENERIC_WRITE;
    share_mode = FILE_SHARE_WRITE;
    protection_flags = PAGE_READWRITE;
  }
  else
  {
    return false;
  }

  LPSECURITY_ATTRIBUTES security_attributes = NULL;
  DWORD creation_disposition = OPEN_EXISTING;
  DWORD flags_and_attributes = FILE_ATTRIBUTE_NORMAL;
  HANDLE file_template = NULL;

  HANDLE file_handle = CreateFileA(
    path,
    desired_access,
    share_mode,
    security_attributes,
    creation_disposition,
    flags_and_attributes,
    file_template
  );

  if (file_handle == INVALID_HANDLE_VALUE)
  {
    return false;
  }

  DWORD maximum_size_high = 0;
  DWORD maximum_size_low = 0;

  HANDLE mapping_handle = CreateFileMappingA(
    file_handle,
    security_attributes,
    protection_flags,
    maximum_size_high,
    maximum_size_low,
    NULL
  );

  if (!mapping_handle)
  {
    CloseHandle(file_handle);
    return false;
  }

  LARGE_INTEGER size;
  if (!GetFileSizeEx(file_handle, &size))
  {
    return false;
  }

  (*file) = new gi_file;
  (*file)->usage = usage;
  (*file)->file_handle = file_handle;
  (*file)->mapping_handle = mapping_handle;
  (*file)->size = (size_t) size.QuadPart;

  return true;
}

size_t gi_file_size(gi_file* file)
{
  return file->size;
}

bool gi_file_close(gi_file* file)
{
  bool closed_mapping = CloseHandle(file->mapping_handle);
  bool closed_file = CloseHandle(file->file_handle);

  delete file;

  return closed_mapping && closed_file;
}

void* gi_mmap(gi_file* file, size_t offset, size_t size)
{
  if (size == 0)
  {
    return NULL;
  }

  DWORD desired_access;
  if (file->usage == GI_FILE_USAGE_WRITE)
  {
    desired_access = FILE_MAP_WRITE;
  }
  else if (file->usage == GI_FILE_USAGE_READ)
  {
    desired_access = FILE_MAP_READ;
  }
  else
  {
    return NULL;
  }

  DWORD file_offset_low = offset & 0x00000000FFFFFFFF;
  DWORD file_offset_high = offset >> 32;

  LPVOID mapped_addr = MapViewOfFile(
    file->mapping_handle,
    desired_access,
    file_offset_high,
    file_offset_low,
    /* This means file sizes greater than 4 GB are not supported on 32-bit systems. */
    (SIZE_T) size
  );

  return mapped_addr;
}

bool gi_munmap(gi_file* file, void* addr)
{
  return UnmapViewOfFile(addr);
}

#else

bool gi_file_create(const char* path, size_t size, gi_file** file)
{
  int open_flags = O_RDWR | O_CREAT | O_TRUNC;
  mode_t permission_flags = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

  int file_descriptor = open(path, open_flags, permission_flags);

  if (file_descriptor < 0)
  {
    return false;
  }

  bool trunc_error = ftruncate(file_descriptor, size);

  if (trunc_error)
  {
    return false;
  }

  (*file) = new gi_file;
  (*file)->usage = GI_FILE_USAGE_WRITE;
  (*file)->file_descriptor = file_descriptor;
  memset((*file)->mapped_ranges, 0, MAX_MAPPED_MEM_RANGES * sizeof(gi_mapped_posix_range));

  return true;
}

bool gi_file_open(const char* path, GiFileUsage usage, gi_file** file)
{
  int open_flags = 0;

  if (usage == GI_FILE_USAGE_WRITE)
  {
    open_flags = O_RDWR;
  }
  else if (usage == GI_FILE_USAGE_READ)
  {
    open_flags = O_RDONLY;
  }
  else
  {
    return false;
  }

  int file_descriptor = open(path, open_flags);

  if (file_descriptor < 0)
  {
    return false;
  }

  struct stat file_stats;
  if (fstat(file_descriptor, &file_stats))
  {
    close(file_descriptor);
    return false;
  }

  (*file) = new gi_file;
  (*file)->usage = usage;
  (*file)->file_descriptor = file_descriptor;
  (*file)->size = file_stats.st_size;
  memset((*file)->mapped_ranges, 0, MAX_MAPPED_MEM_RANGES * sizeof(gi_mapped_posix_range));

  return true;
}

size_t gi_file_size(gi_file* file)
{
  return file->size;
}

bool gi_file_close(gi_file* file)
{
  int result = close(file->file_descriptor);
#ifndef NDEBUG
  /* Make sure all ranges have been unmapped. */
  for (uint32_t i = 0; i < MAX_MAPPED_MEM_RANGES; ++i)
  {
    assert(!file->mapped_ranges[i].addr);
  }
#endif
  delete file;
  return !result;
}

void* gi_mmap(gi_file* file, size_t offset, size_t size)
{
  if (size == 0)
  {
    return NULL;
  }

  /* Try to find an empty mapped range data struct. */
  gi_mapped_posix_range* range = NULL;
  for (uint32_t i = 0; i < MAX_MAPPED_MEM_RANGES; ++i)
  {
    if (!file->mapped_ranges[i].addr)
    {
      range = &file->mapped_ranges[i];
      break;
    }
  }
  if (!range)
  {
    return NULL;
  }

  /* Map the memory. */
  int protection_flags = PROT_READ;

  if (file->usage == GI_FILE_USAGE_WRITE)
  {
    protection_flags |= PROT_WRITE;
  }

  int visibility_flags = MAP_SHARED;
  void* addr = NULL;

  void* mapped_addr = mmap(
    addr,
    size,
    protection_flags,
    visibility_flags,
    file->file_descriptor,
    offset
  );

  if (mapped_addr == MAP_FAILED)
  {
    return NULL;
  }

  range->addr = mapped_addr;
  range->size = size;

  return mapped_addr;
}

bool gi_munmap(gi_file* file, void* addr)
{
  for (uint32_t i = 0; i < MAX_MAPPED_MEM_RANGES; ++i)
  {
    gi_mapped_posix_range* range = &file->mapped_ranges[i];
    if (range->addr != addr)
    {
      continue;
    }
    range->addr = NULL;
    return !munmap(addr, range->size);
  }
  return false;
}

#endif
