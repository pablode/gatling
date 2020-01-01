#include "handle_store.h"

#include <assert.h>

void handle_store_create(
  handle_store* store)
{
  store->max_index = 0;
  store->free_index_count = 0;
  store->free_index_capacity = 8;
  store->free_indices= malloc(store->free_index_count * sizeof(uint32_t));
  store->version_capacity = 8;
  store->versions = malloc(store->version_capacity * sizeof(uint32_t));
}

void handle_store_destroy(
  handle_store* store)
{
  free(store->versions);
  free(store->free_indices);
}

uint64_t handle_store_create_handle(
  handle_store* store)
{
  assert(store->max_index < ~0ul);

  uint32_t version;
  uint32_t index;

  if (store->free_index_count == 0)
  {
    version = 1;
    index = store->max_index++;

    if (index >= store->version_capacity) {
      const uint32_t next_multiple_of_two =
        ((store->version_capacity + 1) / 2) * 2;
      store->versions = realloc(
        store->versions,
        sizeof(uint32_t) * next_multiple_of_two
      );
      store->version_capacity = next_multiple_of_two;
    }
    store->versions[index] = 1;
  }
  else
  {
    index = store->free_indices[store->free_index_count - 1];
    store->free_index_count--;
    version = store->versions[index];
  }

  const uint64_t handle =
    ((uint64_t) index) | (((uint64_t) version) << 32ul);

  return handle;
}

bool handle_store_is_handle_valid(
  const handle_store* store,
  uint64_t handle)
{
  const uint32_t version = (uint32_t) (handle >> 32ul);
  const uint32_t index =   (uint32_t) (handle);
  if (index >= store->max_index) {
    return false;
  }
  uint32_t saved_version = store->versions[index];
  if (saved_version != version) {
    return false;
  }
  return true;
}

void handle_store_free_handle(
  handle_store* store,
  uint64_t handle)
{
  const uint32_t index = handle_store_get_index(handle);
  uint32_t version = store->versions[index];
  version++;
  store->versions[index] = version;
  store->free_index_count++;
  if (store->free_index_count >= store->free_index_capacity) {
    const uint32_t next_multiple_of_two =
        ((store->free_index_count + 1) / 2) * 2;
    store->free_indices = realloc(
        store->free_indices,
        sizeof(uint32_t) * next_multiple_of_two
    );
    store->free_index_capacity = next_multiple_of_two;
  }
  store->free_indices[store->free_index_count - 1] = index;
}

uint32_t handle_store_get_index(
  uint64_t handle)
{
  return (uint32_t) handle;
}
