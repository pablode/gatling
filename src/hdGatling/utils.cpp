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

#include <gtl/gi/Gi.h>

#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>

using namespace gtl;

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
  template <class To>
  struct _TypeConversionHelper {
    template <class From>
    inline To operator()(From const &from) const { return To(from); }
  };
}

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

bool HdGatlingUnboxPRSPrimvars(VtValue& boxedTranslations,
                               VtValue& boxedRotations,
                               VtValue& boxedScales,
                               VtVec3dArray& translations,
                               VtQuatdArray& rotations,
                               VtVec3dArray& scales)
{
  if (boxedTranslations.CanCast<VtVec3dArray>())
  {
    translations = boxedTranslations.Cast<VtVec3dArray>().UncheckedGet<VtVec3dArray>();
  }
  else if (!boxedTranslations.IsEmpty())
  {
    TF_CODING_WARNING("Translation value type %s not supported", boxedTranslations.GetTypeName().c_str());
    return false;
  }

  if (boxedRotations.IsHolding<VtQuatdArray>())
  {
    rotations = boxedRotations.UncheckedGet<VtQuatdArray>();
  }
  else if (boxedRotations.IsHolding<VtQuatfArray>())
  {
    auto& rawArray = boxedRotations.UncheckedGet<VtQuatfArray>();
    rotations.resize(rawArray.size());
    std::transform(rawArray.begin(), rawArray.end(), rotations.begin(), _TypeConversionHelper<GfQuatd>());
  }
  else if (boxedRotations.IsHolding<VtQuathArray>())
  {
    auto& rawArray = boxedRotations.UncheckedGet<VtQuathArray>();
    rotations.resize(rawArray.size());
    std::transform(rawArray.begin(), rawArray.end(), rotations.begin(), _TypeConversionHelper<GfQuatd>());
  }
  else if (!boxedRotations.IsEmpty())
  {
    TF_CODING_WARNING("Rotation value type %s not supported", boxedRotations.GetTypeName().c_str());
    return false;
  }

  if (boxedScales.CanCast<VtVec3dArray>())
  {
    scales = boxedScales.Cast<VtVec3dArray>().UncheckedGet<VtVec3dArray>();
  }
  else if (!boxedScales.IsEmpty())
  {
    TF_CODING_WARNING("Sale value type %s not supported", boxedScales.GetTypeName().c_str());
    return false;
  }

  return true;
}

void HdGatlingPRSToTransforms(const VtIntArray& indices,
                              const GfMatrix4d& rootTransform,
                              const VtMatrix4dArray& instanceTransforms,
                              const VtVec3dArray& translations,
                              const VtQuatdArray& rotations,
                              const VtVec3dArray& scales,
                              VtMatrix4fArray& transforms)
{
  transforms.resize(indices.size());

  for (size_t i = 0; i < indices.size(); i++)
  {
    int instanceIndex = indices[i];

    auto mat = GfMatrix4f(rootTransform);

    GfMatrix4f temp;

    if (i < translations.size())
    {
      auto t = GfVec3f(translations[instanceIndex]);
      temp.SetTranslate(t);
      mat = temp * mat;
    }
    if (i < rotations.size())
    {
      auto r = GfQuatf(rotations[instanceIndex]);
      temp.SetRotate(r);
      mat = temp * mat;
    }
    if (i < scales.size())
    {
      auto s = GfVec3f(scales[instanceIndex]);
      temp.SetScale(s);
      mat = temp * mat;
    }
    if (i < instanceTransforms.size())
    {
      temp = GfMatrix4f(instanceTransforms[instanceIndex]);
      mat = temp * mat;
    }

    transforms[i] = mat;
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
