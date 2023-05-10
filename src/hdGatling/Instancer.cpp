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

#include "Instancer.h"

#include <pxr/imaging/hd/sceneDelegate.h>
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

    if (primName != HdInstancerTokens->translate &&
        primName != HdInstancerTokens->rotate &&
        primName != HdInstancerTokens->scale &&
        primName != HdInstancerTokens->instanceTransform)
    {
      continue;
    }

    if (!HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, primName))
    {
      continue;
    }

    VtValue value = sceneDelegate->Get(id, primName);

    m_primvarMap[primName] = value;
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

VtMatrix4dArray HdGatlingInstancer::ComputeInstanceTransforms(const SdfPath& prototypeId)
{
  HdSceneDelegate* sceneDelegate = GetDelegate();

  const SdfPath& id = GetId();

  // Calculate instance transforms for this instancer.
  VtValue boxedTranslates = m_primvarMap[HdInstancerTokens->translate];
  VtValue boxedRotates = m_primvarMap[HdInstancerTokens->rotate];
  VtValue boxedScales = m_primvarMap[HdInstancerTokens->scale];
  VtValue boxedInstanceTransforms = m_primvarMap[HdInstancerTokens->instanceTransform];

  VtVec3dArray translates;
  if (boxedTranslates.CanCast<VtVec3dArray>())
  {
    translates = boxedTranslates.Cast<VtVec3dArray>().UncheckedGet<VtVec3dArray>();
  }
  else if (!boxedTranslates.IsEmpty())
  {
    TF_CODING_WARNING("Instancer translate value type %s not supported", boxedTranslates.GetTypeName().c_str());
  }

  VtQuatdArray rotates;
  if (boxedRotates.IsHolding<VtQuatdArray>())
  {
    rotates = boxedRotates.UncheckedGet<VtQuatdArray>();
  }
  else if (boxedRotates.IsHolding<VtQuatfArray>())
  {
    auto& rawArray = boxedRotates.UncheckedGet<VtQuatfArray>();
    rotates.resize(rawArray.size());
    std::transform(rawArray.begin(), rawArray.end(), rotates.begin(), _TypeConversionHelper<GfQuatd>());
  }
  else if (boxedRotates.IsHolding<VtQuathArray>())
  {
    auto& rawArray = boxedRotates.UncheckedGet<VtQuathArray>();
    rotates.resize(rawArray.size());
    std::transform(rawArray.begin(), rawArray.end(), rotates.begin(), _TypeConversionHelper<GfQuatd>());
  }
  else if (!boxedRotates.IsEmpty())
  {
    TF_CODING_WARNING("Instancer rotate value type %s not supported", boxedRotates.GetTypeName().c_str());
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

  VtMatrix4dArray transforms;
  transforms.resize(instanceIndices.size());

  for (size_t i = 0; i < instanceIndices.size(); i++)
  {
    int instanceIndex = instanceIndices[i];

    GfMatrix4d mat = instancerTransform;

    GfMatrix4d temp;

    if (i < translates.size())
    {
      temp.SetTranslate(translates[instanceIndex]);
      mat = temp * mat;
    }
    if (i < rotates.size())
    {
      temp.SetRotate(rotates[instanceIndex]);
      mat = temp * mat;
    }
    if (i < scales.size())
    {
      temp.SetScale(scales[instanceIndex]);
      mat = temp * mat;
    }
    if (i < instanceTransforms.size())
    {
      temp = instanceTransforms[instanceIndex];
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

  VtMatrix4dArray parentTransforms = parentInstancer->ComputeInstanceTransforms(id);

  VtMatrix4dArray transformProducts;
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
