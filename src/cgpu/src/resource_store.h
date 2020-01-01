#ifndef CGPU_RESOURCE_STORE_H
#define CGPU_RESOURCE_STORE_H

#include <stdlib.h>

#include "handle_store.h"

typedef struct resource_store {
  handle_store handle_store;
  void*        objects;
  size_t       object_count;
  size_t       item_byte_size;
} resource_store;

void resource_store_create(
  resource_store* store,
  size_t item_byte_size,
  size_t initial_capacity
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
