#ifndef GATLING_MMAP_H
#define GATLING_MMAP_H

#include <stdbool.h>
#include <stdint.h>

typedef struct gatling_file gatling_file;

typedef enum GatlingFileUsage {
  GATLING_FILE_USAGE_READ  = 1,
  GATLING_FILE_USAGE_WRITE = 2
} GatlingFileUsage;

bool gatling_file_create(
  const char* path,
  uint64_t byte_count,
  gatling_file** file
);

bool gatling_file_open(
  const char* path,
  GatlingFileUsage usage,
  gatling_file** file
);

uint64_t gatling_file_size(gatling_file* file);

bool gatling_file_close(gatling_file* file);

void* gatling_mmap(
  gatling_file* file,
  uint64_t byte_offset,
  uint64_t byte_count
);

bool gatling_munmap(
  gatling_file* file,
  void* addr
);

#endif
