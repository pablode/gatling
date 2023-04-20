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

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

struct handle_store
{
  uint32_t  max_index;
  uint32_t* versions;
  uint32_t  version_capacity;
  uint32_t* free_indices;
  uint32_t  free_index_count;
  uint32_t  free_index_capacity;
};

void handle_store_create(handle_store* store);

void handle_store_destroy(handle_store* store);

uint64_t handle_store_create_handle(handle_store* store);

bool handle_store_is_handle_valid(const handle_store* store, uint64_t handle);

void handle_store_free_handle(handle_store* store, uint64_t handle);

uint32_t handle_store_get_index(uint64_t handle);
