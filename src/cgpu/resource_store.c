#include "resource_store.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

void resource_store_create(
    resource_store* store,
    size_t item_byte_size,
    size_t initial_capacity)
{
  assert(initial_capacity != 0);
  handle_store_create(&store->handle_store);
  store->objects = NULL;
  store->object_count = 0;

  const size_t ptr_size = sizeof(void*);
  store->item_byte_size = (item_byte_size + ptr_size - 1) / ptr_size * ptr_size;
  store->objects = malloc(store->item_byte_size * initial_capacity);
  store->object_count = initial_capacity;
}

void resource_store_destroy(resource_store* store)
{
  handle_store_destroy(&store->handle_store);
  free(store->objects);
}

uint64_t resource_store_create_handle(resource_store* store)
{
  return handle_store_create_handle(&store->handle_store);
}

void resource_store_free_handle(resource_store* store, uint64_t handle)
{
  handle_store_free_handle(&store->handle_store, handle);
}

bool resource_store_get(resource_store* store, uint64_t handle, void** object)
{
#ifndef NDEBUG
  if (!handle_store_is_handle_valid(&store->handle_store, handle)) {
    return false;
  }
#endif

  const uint32_t index = handle_store_get_index(handle);
  if (index >= store->object_count) {
    const size_t new_count = store->object_count * 2;
    store->objects = realloc(
        store->objects,
        new_count * store->item_byte_size
    );
    store->object_count = new_count;
  }

  *object =  store->objects + store->item_byte_size * index;
  return true;
}
