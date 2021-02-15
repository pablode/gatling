#ifndef CGPU_HANDLE_STORE_H
#define CGPU_HANDLE_STORE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct handle_store {
  uint32_t  max_index;
  uint32_t* versions;
  uint32_t  version_capacity;
  uint32_t* free_indices;
  uint32_t  free_index_count;
  uint32_t  free_index_capacity;
} handle_store;

void handle_store_create(
  handle_store* store
);

void handle_store_destroy(
  handle_store* store
);

uint64_t handle_store_create_handle(
  handle_store* store
);

bool handle_store_is_handle_valid(
  const handle_store* store,
  uint64_t handle
);

void handle_store_free_handle(
  handle_store* store,
  uint64_t handle
);

uint32_t handle_store_get_index(
  uint64_t handle
);

#endif
