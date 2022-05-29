/*
 * Copyright (C) 2019-2022 Pablo Delgado Krämer
 *
 * This program is free software: you can redistribute it and/or modify
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
