//
// Copyright (C) 2023 Pablo Delgado Krämer
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

#include "handle_store.h"

#include <assert.h>

void handle_store_create(handle_store* store)
{
  store->max_index = 0;
  store->free_index_count = 0;
  store->free_index_capacity = 8;
  store->free_indices= (uint32_t*) malloc(store->free_index_capacity * sizeof(uint32_t));
  store->version_capacity = 8;
  store->versions = (uint32_t*) malloc(store->version_capacity * sizeof(uint32_t));
}

void handle_store_destroy(handle_store* store)
{
  free(store->versions);
  free(store->free_indices);
}

// See: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
uint32_t handle_store_next_power_of_two(uint32_t v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

uint64_t handle_store_create_handle(handle_store* store)
{
  assert(store->max_index < (~0u));

  uint32_t index;
  uint32_t version;

  if (store->free_index_count > 0)
  {
    index = store->free_indices[store->free_index_count - 1];
    store->free_index_count--;
    version = store->versions[index];
  }
  else
  {
    index = store->max_index++;
    version = 1;

    if (index >= store->version_capacity)
    {
      store->version_capacity = handle_store_next_power_of_two(index + 1);
      store->versions = (uint32_t*) realloc(store->versions, sizeof(uint32_t) * store->version_capacity);
    }

    store->versions[index] = version;
  }

  uint64_t handle = ((uint64_t) index) | (((uint64_t) version) << 32ul);

  return handle;
}

bool handle_store_is_handle_valid(const handle_store* store, uint64_t handle)
{
  uint32_t version = (uint32_t) (handle >> 32ul);
  uint32_t index = (uint32_t) (handle);
  if (index > store->max_index)
  {
    return false;
  }

  uint32_t saved_version = store->versions[index];
  if (saved_version != version)
  {
    return false;
  }

  return true;
}

void handle_store_free_handle(handle_store* store, uint64_t handle)
{
  uint32_t index = handle_store_get_index(handle);
  store->versions[index]++;
  store->free_index_count++;

  if (store->free_index_count > store->free_index_capacity)
  {
    store->free_index_capacity = handle_store_next_power_of_two(store->free_index_count);
    store->free_indices = (uint32_t*) realloc(store->free_indices, sizeof(uint32_t) * store->free_index_capacity);
  }

  store->free_indices[store->free_index_count - 1] = index;
}

uint32_t handle_store_get_index(uint64_t handle)
{
  return (uint32_t) handle;
}
