//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#define GB_DECLARE_ENUM_BITOPS(ENUMTYPE) \
  inline constexpr ENUMTYPE operator | (ENUMTYPE a, ENUMTYPE b) \
  { return ENUMTYPE((int) a | (int) b); }                       \
  inline ENUMTYPE& operator |= (ENUMTYPE& a, ENUMTYPE  b)       \
  { return (ENUMTYPE&) (((int&) a) |= ((int) b)); }             \
  inline constexpr ENUMTYPE operator & (ENUMTYPE a, ENUMTYPE b) \
  { return ENUMTYPE(((int) a) & ((int) b)); }                   \
  inline ENUMTYPE& operator &= (ENUMTYPE &a, ENUMTYPE b)        \
  { return (ENUMTYPE&) (((int&) a) &= ((int) b)); }             \
  inline constexpr ENUMTYPE operator ~ (ENUMTYPE a)             \
  { return ENUMTYPE(~((int) a)); }                              \
  inline constexpr ENUMTYPE operator ^ (ENUMTYPE a, ENUMTYPE b) \
  { return ENUMTYPE(((int) a) ^ ((int) b)); }                   \
  inline ENUMTYPE& operator ^= (ENUMTYPE &a, ENUMTYPE b)        \
  { return (ENUMTYPE&) (((int&) a) ^= ((int) b)); }
