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

namespace
{
  constexpr static const uint32_t MAX_MAPPED_MEM_RANGES = 16;

  struct _MappedPosixRange
  {
    void* addr;
    size_t size;
  };
}

namespace gtl
{
  struct GiFile
  {
    GiFileUsage usage;
    size_t      size;
#if defined(_WIN32)
    HANDLE      fileHandle;
    HANDLE      mappingHandle;
#else
    int         fileDescriptor;
    _MappedPosixRange mappedRanges[MAX_MAPPED_MEM_RANGES];
#endif
  };

#if defined (_WIN32)

  bool giFileCreate(const char* path, size_t size, GiFile** file)
  {
    DWORD desiredAccess = GENERIC_READ | GENERIC_WRITE;
    DWORD shareMode = FILE_SHARE_WRITE;
    LPSECURITY_ATTRIBUTES securityAttributes = nullptr;
    DWORD creationDisposition = CREATE_ALWAYS;
    DWORD flagsAndAttributes = FILE_ATTRIBUTE_NORMAL;
    HANDLE fileTemplate = nullptr;

    HANDLE fileHandle = CreateFileA(
      path,
      desiredAccess,
      shareMode,
      securityAttributes,
      creationDisposition,
      flagsAndAttributes,
      fileTemplate
    );

    if (fileHandle == INVALID_HANDLE_VALUE)
    {
      return false;
    }

    DWORD protectionFlags = PAGE_READWRITE;
    DWORD maximumSizeHigh = size >> 32;
    DWORD maximumSizeLow = size & 0x00000000FFFFFFFF;

    /* "If an application specifies a size for the file mapping object that is
     * larger than the size of the actual named file on disk and if the page
     * protection allows write access, then the file on disk is increased to
     * match the specified size of the file mapping object." (MSDN, 2020-04-09) */
    HANDLE mappingHandle = CreateFileMappingA(
      fileHandle,
      securityAttributes,
      protectionFlags,
      maximumSizeHigh,
      maximumSizeLow,
      nullptr
    );

    if (!mappingHandle)
    {
      CloseHandle(fileHandle);
      return false;
    }

    (*file) = new GiFile;
    (*file)->usage = GiFileUsage::Write;
    (*file)->fileHandle = fileHandle;
    (*file)->mappingHandle = mappingHandle;
    (*file)->size = size;

    return true;
  }

  bool giFileOpen(const char* path, GiFileUsage usage, GiFile** file)
  {
    DWORD desiredAccess;
    DWORD shareMode;
    DWORD protectionFlags;
    if (usage == GiFileUsage::Read)
    {
      desiredAccess = GENERIC_READ;
      shareMode = FILE_SHARE_READ;
      protectionFlags = PAGE_READONLY;
    }
    else if (usage == GiFileUsage::Write)
    {
      desiredAccess = GENERIC_READ | GENERIC_WRITE;
      shareMode = FILE_SHARE_WRITE;
      protectionFlags = PAGE_READWRITE;
    }
    else
    {
      return false;
    }

    LPSECURITY_ATTRIBUTES securityAttributes = nullptr;
    DWORD creationDisposition = OPEN_EXISTING;
    DWORD flagsAndAttributes = FILE_ATTRIBUTE_NORMAL;
    HANDLE fileTemplate = nullptr;

    HANDLE fileHandle = CreateFileA(
      path,
      desiredAccess,
      shareMode,
      securityAttributes,
      creationDisposition,
      flagsAndAttributes,
      fileTemplate
    );

    if (fileHandle == INVALID_HANDLE_VALUE)
    {
      return false;
    }

    DWORD maximumSizeHigh = 0;
    DWORD maximumSizeLow = 0;

    HANDLE mappingHandle = CreateFileMappingA(
      fileHandle,
      securityAttributes,
      protectionFlags,
      maximumSizeHigh,
      maximumSizeLow,
      nullptr
    );

    if (!mappingHandle)
    {
      CloseHandle(fileHandle);
      return false;
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(fileHandle, &size))
    {
      return false;
    }

    (*file) = new GiFile;
    (*file)->usage = usage;
    (*file)->fileHandle = fileHandle;
    (*file)->mappingHandle = mappingHandle;
    (*file)->size = (size_t)size.QuadPart;

    return true;
  }

  size_t giFileSize(GiFile* file)
  {
    return file->size;
  }

  bool giFileClose(GiFile* file)
  {
    bool closedMapping = CloseHandle(file->mappingHandle);
    bool closedFile = CloseHandle(file->fileHandle);

    delete file;

    return closedMapping && closedFile;
  }

  void* giMmap(GiFile* file, size_t offset, size_t size)
  {
    if (size == 0)
    {
      return nullptr;
    }

    DWORD desiredAccess;
    if (file->usage == GiFileUsage::Write)
    {
      desiredAccess = FILE_MAP_WRITE;
    }
    else if (file->usage == GiFileUsage::Read)
    {
      desiredAccess = FILE_MAP_READ;
    }
    else
    {
      return nullptr;
    }

    DWORD fileOffsetHigh = offset >> 32;
    DWORD fileOffsetLow = offset & 0x00000000FFFFFFFF;

    LPVOID mappedAddr = MapViewOfFile(
      file->mappingHandle,
      desiredAccess,
      fileOffsetHigh,
      fileOffsetLow,
      /* This means file sizes greater than 4 GB are not supported on 32-bit systems. */
      (SIZE_T)size
    );

    return mappedAddr;
  }

  bool giMunmap(GiFile* file, void* addr)
  {
    return UnmapViewOfFile(addr);
  }

#else

  bool giFileCreate(const char* path, size_t size, GiFile** file)
  {
    int openFlags = O_RDWR | O_CREAT | O_TRUNC;
    mode_t permissionFlags = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

    int fileDescriptor = open(path, openFlags, permissionFlags);

    if (fileDescriptor < 0)
    {
      return false;
    }

    bool truncError = ftruncate(fileDescriptor, size);

    if (truncError)
    {
      return false;
    }

    (*file) = new GiFile;
    (*file)->usage = GiFileUsage::Write;
    (*file)->fileDescriptor = fileDescriptor;
    memset((*file)->mappedRanges, 0, MAX_MAPPED_MEM_RANGES * sizeof(_MappedPosixRange));

    return true;
  }

  bool giFileOpen(const char* path, GiFileUsage usage, GiFile** file)
  {
    int openFlags = 0;

    if (usage == GiFileUsage::Write)
    {
      openFlags = O_RDWR;
    }
    else if (usage == GiFileUsage::Read)
    {
      openFlags = O_RDONLY;
    }
    else
    {
      return false;
    }

    int fileDescriptor = open(path, openFlags);

    if (fileDescriptor < 0)
    {
      return false;
    }

    struct stat fileStats;
    if (fstat(fileDescriptor, &fileStats))
    {
      close(fileDescriptor);
      return false;
    }

    (*file) = new GiFile;
    (*file)->usage = usage;
    (*file)->fileDescriptor = fileDescriptor;
    (*file)->size = fileStats.st_size;
    memset((*file)->mappedRanges, 0, MAX_MAPPED_MEM_RANGES * sizeof(_MappedPosixRange));

    return true;
  }

  size_t giFileSize(GiFile* file)
  {
    return file->size;
  }

  bool giFileClose(GiFile* file)
  {
    int result = close(file->fileDescriptor);
#ifndef NDEBUG
    /* Make sure all ranges have been unmapped. */
    for (uint32_t i = 0; i < MAX_MAPPED_MEM_RANGES; ++i)
    {
      assert(!file->mappedRanges[i].addr);
    }
#endif
    delete file;
    return !result;
  }

  void* giMmap(GiFile* file, size_t offset, size_t size)
  {
    if (size == 0)
    {
      return nullptr;
    }

    /* Try to find an empty mapped range data struct. */
    _MappedPosixRange* range = nullptr;
    for (uint32_t i = 0; i < MAX_MAPPED_MEM_RANGES; ++i)
    {
      if (!file->mappedRanges[i].addr)
      {
        range = &file->mappedRanges[i];
        break;
      }
    }
    if (!range)
    {
      return nullptr;
    }

    /* Map the memory. */
    int protectionFlags = PROT_READ;

    if (file->usage == GiFileUsage::Write)
    {
      protectionFlags |= PROT_WRITE;
    }

    int visibilityFlags = MAP_SHARED;
    void* addr = nullptr;

    void* mappedAddr = mmap(
      addr,
      size,
      protectionFlags,
      visibilityFlags,
      file->fileDescriptor,
      offset
    );

    if (mappedAddr == MAP_FAILED)
    {
      return nullptr;
    }

    range->addr = mappedAddr;
    range->size = size;

    return mappedAddr;
  }

  bool giMunmap(GiFile* file, void* addr)
  {
    for (uint32_t i = 0; i < MAX_MAPPED_MEM_RANGES; ++i)
    {
      _MappedPosixRange* range = &file->mappedRanges[i];
      if (range->addr != addr)
      {
        continue;
      }
      range->addr = nullptr;
      return !munmap(addr, range->size);
    }
    return false;
  }

#endif
}
