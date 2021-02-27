#include "Instancer.h"

#include <pxr/imaging/hd/sceneDelegate.h>
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

VtMatrix4dArray HdGatlingInstancer::ComputeInstanceTransforms(const SdfPath& prototypeId)
{
  HdSceneDelegate* sceneDelegate = GetDelegate();

  const SdfPath& id = GetId();

  // Calculate instance transforms for this instancer.
  VtValue boxedTranslates = m_primvarMap[HdInstancerTokens->translate];
  VtValue boxedRotates = m_primvarMap[HdInstancerTokens->rotate];
  VtValue boxedScales = m_primvarMap[HdInstancerTokens->scale];
  VtValue boxedInstanceTransforms = m_primvarMap[HdInstancerTokens->instanceTransform];

  VtVec3fArray translates;
  if (boxedTranslates.IsHolding<VtVec3fArray>())
  {
    translates = boxedTranslates.UncheckedGet<VtVec3fArray>();
  }
  else if (!boxedTranslates.IsEmpty())
  {
    TF_CODING_WARNING("Instancer translate values are not of type Vec3f!");
  }

  VtVec4fArray rotates;
  if (boxedRotates.IsHolding<VtVec4fArray>())
  {
    rotates = boxedRotates.Get<VtVec4fArray>();
  }
  else if (!boxedRotates.IsEmpty())
  {
    TF_CODING_WARNING("Instancer rotate values are not of type Vec3f!");
  }

  VtVec3fArray scales;
  if (boxedScales.IsHolding<VtVec3fArray>())
  {
    scales = boxedScales.Get<VtVec3fArray>();
  }
  else if (!boxedScales.IsEmpty())
  {
    TF_CODING_WARNING("Instancer scale values are not of type Vec3f!");
  }

  VtMatrix4dArray instanceTransforms;
  if (boxedInstanceTransforms.IsHolding<VtMatrix4dArray>())
  {
    instanceTransforms = boxedInstanceTransforms.Get<VtMatrix4dArray>();
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
      auto trans = GfVec3d(translates[instanceIndex]);
      temp.SetTranslate(trans);
      mat = temp * mat;
    }
    if (i < rotates.size())
    {
      GfVec4f rot = rotates[instanceIndex];
      temp.SetRotate(GfQuatd(rot[0], rot[1], rot[2], rot[3]));
      mat = temp * mat;
    }
    if (i < scales.size())
    {
      auto scale = GfVec3d(scales[instanceIndex]);
      temp.SetScale(scale);
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
  HdGatlingInstancer* parentInstancer = dynamic_cast<HdGatlingInstancer*>(boxedParentInstancer);

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
