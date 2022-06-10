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

#ifndef CGPU_SHADER_REFLECTION_H
#define CGPU_SHADER_REFLECTION_H

#include "cgpu.h"

typedef struct cgpu_shader_reflection_binding {
  uint32_t binding;
  int descriptor_type;
  bool write_access;
  bool read_access;
  uint32_t count;
} cgpu_shader_reflection_binding;

typedef struct cgpu_shader_reflection {
  uint32_t push_constants_size;
  uint32_t binding_count;
  cgpu_shader_reflection_binding* bindings;
} cgpu_shader_reflection;

bool cgpu_perform_shader_reflection(
  uint64_t size,
  const uint32_t* p_spv,
  cgpu_shader_reflection* p_reflection
);

void cgpu_destroy_shader_reflection(
  cgpu_shader_reflection* p_reflection
);

#endif
