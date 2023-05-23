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

#ifndef SI_GTL
#define SI_GTL

#ifdef __cplusplus

#include <glm/glm.hpp>

#define SI_INT        int32_t
#define SI_UINT       uint32_t
#define SI_FLOAT      float
#define SI_VEC2       glm::vec2
#define SI_VEC3       glm::vec3
#define SI_VEC4       glm::vec4
#define SI_MAT3       glm::mat3
#define SI_MAT3x4     glm::mat3x4

#define SI_NAMESPACE_BEGIN(NAME) \
  namespace gtl {                \
    namespace shader_interface { \
      namespace NAME {

#define SI_NAMESPACE_END()       \
      }                          \
    }                            \
  }

#define SI_BINDING_INDEX(NAME, IDX) \
  constexpr static uint32_t BINDING_INDEX_##NAME = IDX;

#else

#define SI_INT        int
#define SI_UINT       uint
#define SI_FLOAT      float
#define SI_VEC2       vec2
#define SI_VEC3       vec3
#define SI_VEC4       vec4
#define SI_MAT3       mat3
#define SI_MAT3x4     mat3x4

#define SI_NAMESPACE_BEGIN(NAME)
#define SI_NAMESPACE_END()

#define SI_BINDING_INDEX(NAME,IDX) \
  const uint BINDING_INDEX_##NAME = IDX;

#endif

#endif
