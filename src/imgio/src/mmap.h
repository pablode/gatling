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
