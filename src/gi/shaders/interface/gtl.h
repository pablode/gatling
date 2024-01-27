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

#ifndef GTL_H
#define GTL_H

#ifdef __cplusplus

#include <glm/glm.hpp>

namespace gtl
{
  namespace shader_interface
  {
    using GI_INT    = int32_t;
    using GI_UINT   = uint32_t;
    using GI_UINT64 = uint64_t;
    using GI_FLOAT  = float;
    using GI_VEC2   = glm::vec2;
    using GI_VEC3   = glm::vec3;
    using GI_VEC4   = glm::vec4;
    using GI_IVEC2  = glm::ivec2;
    using GI_IVEC3  = glm::ivec3;
    using GI_IVEC4  = glm::ivec4;
    using GI_UVEC2  = glm::uvec2;
    using GI_UVEC3  = glm::uvec3;
    using GI_UVEC4  = glm::uvec4;
    using GI_MAT3   = glm::mat3;
    using GI_MAT3x4 = glm::mat3x4;
  }
}

#define GI_INTERFACE_BEGIN(NAME)   \
  namespace gtl {                  \
    namespace shader_interface {   \
      namespace NAME {

#define GI_INTERFACE_END()   \
      }                      \
    }                        \
  }

#define GI_BINDING_INDEX(NAME, IDX)   \
  constexpr static uint32_t BINDING_INDEX_##NAME = IDX;

#else

#define GI_INT        int
#define GI_UINT       uint
#define GI_UINT64     uint64_t
#define GI_FLOAT      float
#define GI_VEC2       vec2
#define GI_VEC3       vec3
#define GI_VEC4       vec4
#define GI_iVEC2      ivec2
#define GI_iVEC3      ivec3
#define GI_iVEC4      ivec4
#define GI_UVEC2      uvec2
#define GI_UVEC3      uvec3
#define GI_UVEC4      uvec4
#define GI_MAT3       mat3
#define GI_MAT3x4     mat3x4

#define GI_INTERFACE_BEGIN(NAME)
#define GI_INTERFACE_END()

#define GI_BINDING_INDEX(NAME,IDX) \
  const uint BINDING_INDEX_##NAME = IDX;

#endif

#endif
