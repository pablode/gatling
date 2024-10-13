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

#include "instancer.h"

#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quatd.h>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingInstancer::HdGatlingInstancer(HdSceneDelegate* delegate,
                                       const SdfPath& id)
  : HdInstancer(delegate, id)
{
}

HdGatlingInstancer::~HdGatlingInstancer()
{
}

void HdGatlingInstancer::Sync(HdSceneDelegate* sceneDelegate,
                              HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits)
{
  TF_UNUSED(renderParam);

  _UpdateInstancer(sceneDelegate, dirtyBits);

  const SdfPath& id = GetId();

  if (!HdChangeTracker::IsAnyPrimvarDirty(*dirtyBits, id))
  {
    return;
  }

  const HdPrimvarDescriptorVector& primvars = sceneDelegate->GetPrimvarDescriptors(id, HdInterpolation::HdInterpolationInstance);

  for (const HdPrimvarDescriptor& primvar : primvars)
  {
    TfToken primName = primvar.name;

    if (!HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, primName))
    {
      continue;
    }

    bool doesPrimvarAffectInstances =
#if PXR_VERSION <= 2308
      primName == HdInstancerTokens->translate || primName == HdInstancerTokens->rotate ||
      primName == HdInstancerTokens->scale || primName == HdInstancerTokens->instanceTransform;
#else
      primName == HdInstancerTokens->instanceTranslations || primName == HdInstancerTokens->instanceRotations ||
      primName == HdInstancerTokens->instanceScales || primName == HdInstancerTokens->instanceTransforms;
#endif

    if (!doesPrimvarAffectInstances)
    {
      continue;
    }

    VtValue value = sceneDelegate->Get(id, primName);

    _primvarMap[primName] = value;
  }
}

namespace
{
  template <class To>
  struct _TypeConversionHelper {
    template <class From>
    inline To operator()(From const &from) const { return To(from); }
  };
}

VtMatrix4fArray HdGatlingInstancer::ComputeInstanceTransforms(const SdfPath& prototypeId)
{
  HdSceneDelegate* sceneDelegate = GetDelegate();

  const SdfPath& id = GetId();

  // Calculate instance transforms for this instancer.
#if PXR_VERSION <= 2308
  VtValue boxedTranslations = _primvarMap[HdInstancerTokens->translate];
  VtValue boxedRotations = _primvarMap[HdInstancerTokens->rotate];
  VtValue boxedScales = _primvarMap[HdInstancerTokens->scale];
  VtValue boxedInstanceTransforms = _primvarMap[HdInstancerTokens->instanceTransform];
#else
  VtValue boxedTranslations = _primvarMap[HdInstancerTokens->instanceTranslations];
  VtValue boxedRotations = _primvarMap[HdInstancerTokens->instanceRotations];
  VtValue boxedScales = _primvarMap[HdInstancerTokens->instanceScales];
  VtValue boxedInstanceTransforms = _primvarMap[HdInstancerTokens->instanceTransforms];
#endif

  VtVec3dArray translations;
  if (boxedTranslations.CanCast<VtVec3dArray>())
  {
    translations = boxedTranslations.Cast<VtVec3dArray>().UncheckedGet<VtVec3dArray>();
  }
  else if (!boxedTranslations.IsEmpty())
  {
    TF_CODING_WARNING("Instancer translate value type %s not supported", boxedTranslations.GetTypeName().c_str());
  }

  VtQuatdArray rotations;
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
    TF_CODING_WARNING("Instancer rotate value type %s not supported", boxedRotations.GetTypeName().c_str());
  }

  VtVec3dArray scales;
  if (boxedScales.CanCast<VtVec3dArray>())
  {
    scales = boxedScales.Cast<VtVec3dArray>().UncheckedGet<VtVec3dArray>();
  }
  else if (!boxedScales.IsEmpty())
  {
    TF_CODING_WARNING("Instancer scale value type %s not supported", boxedScales.GetTypeName().c_str());
  }

  VtMatrix4dArray instanceTransforms;
  if (boxedInstanceTransforms.CanCast<VtMatrix4dArray>())
  {
    instanceTransforms = boxedInstanceTransforms.UncheckedGet<VtMatrix4dArray>();
  }

  GfMatrix4d instancerTransform = sceneDelegate->GetInstancerTransform(id);

  const VtIntArray& instanceIndices = sceneDelegate->GetInstanceIndices(id, prototypeId);

  VtMatrix4fArray transforms;
  transforms.resize(instanceIndices.size());

  for (size_t i = 0; i < instanceIndices.size(); i++)
  {
    int instanceIndex = instanceIndices[i];

    auto mat = GfMatrix4f(instancerTransform);

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

  // Calculate instance transforms for all instancer instances.
  const SdfPath& parentId = GetParentId();

  if (parentId.IsEmpty())
  {
    return transforms;
  }

  const HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();
  HdInstancer* boxedParentInstancer = renderIndex.GetInstancer(parentId);
  HdGatlingInstancer* parentInstancer = static_cast<HdGatlingInstancer*>(boxedParentInstancer);

  VtMatrix4fArray parentTransforms = parentInstancer->ComputeInstanceTransforms(id);

  VtMatrix4fArray transformProducts;
  transformProducts.resize(parentTransforms.size() * transforms.size());

  for (size_t i = 0; i < parentTransforms.size(); i++)
  {
    for (size_t j = 0; j < transforms.size(); j++)
    {
      size_t index = i * transforms.size() + j;

      transformProducts[index] = transforms[j] * parentTransforms[i];
    }
  }

  return transformProducts;
}

PXR_NAMESPACE_CLOSE_SCOPE
