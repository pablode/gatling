#ifndef CGPU_RESOURCE_STORE_H
#define CGPU_RESOURCE_STORE_H

#include <stdlib.h>

#include "handle_store.h"

typedef struct resource_store {
  handle_store handle_store;
  void*        objects;
  uint32_t     object_count;
  uint32_t     item_byte_size;
} resource_store;

void resource_store_create(
  resource_store* store,
  uint32_t item_byte_size,
  uint32_t initial_capacity
);

void resource_store_destroy(
  resource_store* store
);

uint64_t resource_store_create_handle(
  resource_store* store
);

void resource_store_free_handle(
  resource_store* store,
  uint64_t handle
);

bool resource_store_get(
  resource_store* store,
  uint64_t handle,
  void** object
);

#endif
