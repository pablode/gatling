//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
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

#include "utils.h"

#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>

using namespace gtl;

PXR_NAMESPACE_OPEN_SCOPE

bool HdGatlingIsPrimvarTypeSupported(const VtValue& value)
{
  return value.IsHolding<VtVec4fArray>() ||
    value.IsHolding<VtVec3fArray>() ||
    value.IsHolding<VtVec2fArray>() ||
    value.IsHolding<VtFloatArray>() ||
    value.IsHolding<VtVec4iArray>() ||
    value.IsHolding<VtVec3iArray>() ||
    value.IsHolding<VtVec2iArray>() ||
    value.IsHolding<VtBoolArray>() ||
    value.IsHolding<VtIntArray>();
}

GiPrimvarType HdGatlingGetGiPrimvarType(HdType type)
{
  switch (type)
  {
  case HdTypeFloat:
    return GiPrimvarType::Float;
  case HdTypeFloatVec2:
    return GiPrimvarType::Vec2;
  case HdTypeFloatVec3:
    return GiPrimvarType::Vec3;
  case HdTypeFloatVec4:
    return GiPrimvarType::Vec4;
  case HdTypeInt32:
    return GiPrimvarType::Int;
  case HdTypeInt32Vec2:
    return GiPrimvarType::Int2;
  case HdTypeInt32Vec3:
    return GiPrimvarType::Int3;
  case HdTypeInt32Vec4:
    return GiPrimvarType::Int4;
  default:
    TF_CODING_ERROR("primvar type %i unsupported", int(type));
    return GiPrimvarType::Float;
  }
}

void HdGatlingConvertVtBoolArrayToVtIntArray(VtValue& values)
{
  auto boolArray = values.Get<VtBoolArray>();
  VtIntArray intArray(boolArray.size());

  for (int i = 0; i < boolArray.size(); i++)
  {
    intArray[i] = boolArray[i] ? 1 : 0;
  }

  values = std::move(intArray);
}

PXR_NAMESPACE_CLOSE_SCOPE
