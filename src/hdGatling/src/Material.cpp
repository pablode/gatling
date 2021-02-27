#include "Material.h"

#include <pxr/usdImaging/usdImaging/tokens.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
  _usdPreviewSurfaceParamTokens,
  (diffuseColor)
  (emissiveColor)
);

HdGatlingMaterial::HdGatlingMaterial(const SdfPath& id)
  : HdMaterial(id)
{
  m_material.albedo[0] = 0.0f;
  m_material.albedo[1] = 0.0f;
  m_material.albedo[2] = 0.0f;
  m_material.emission[0] = 0.0f;
  m_material.emission[1] = 0.0f;
  m_material.emission[2] = 0.0f;
}

HdGatlingMaterial::~HdGatlingMaterial()
{
}

const gi_material& HdGatlingMaterial::GetGiMaterial() const
{
  return m_material;
}

void HdGatlingMaterial::Sync(HdSceneDelegate* sceneDelegate,
                             HdRenderParam* renderParam,
                             HdDirtyBits* dirtyBits)
{
  TF_UNUSED(renderParam);

  bool pullMaterial = (*dirtyBits & DirtyBits::DirtyParams);

  *dirtyBits = DirtyBits::Clean;

  if (!pullMaterial)
  {
    return;
  }

  const SdfPath& id = GetId();

  VtValue resource = sceneDelegate->GetMaterialResource(id);

  if (!resource.IsHolding<HdMaterialNetworkMap>())
  {
    return;
  }

  const HdMaterialNetworkMap& networkMap = resource.UncheckedGet<HdMaterialNetworkMap>();

  const HdMaterialNetwork* network = TfMapLookupPtr(networkMap.map, HdMaterialTerminalTokens->surface);

  if (!network)
  {
    return;
  }

  _ReadMaterialNetwork(network);
}

void HdGatlingMaterial::_ReadMaterialNetwork(const HdMaterialNetwork* network)
{
  // Instead of actually trying to understand different nodes and the connections
  // between them, we just greedily fill our data from all UsdPreviewSurface nodes.
  for (const HdMaterialNode& node : network->nodes)
  {
    if (node.identifier != UsdImagingTokens->UsdPreviewSurface)
    {
      continue;
    }

    auto boxedDiffuseColorIter = node.parameters.find(_usdPreviewSurfaceParamTokens->diffuseColor);
    auto boxedEmissiveColorIter = node.parameters.find(_usdPreviewSurfaceParamTokens->emissiveColor);

    if (boxedDiffuseColorIter != node.parameters.end())
    {
      VtValue vtValue = boxedDiffuseColorIter->second;;
      GfVec3f diffuseColor = vtValue.Get<GfVec3f>();

      m_material.albedo[0] = diffuseColor[0];
      m_material.albedo[1] = diffuseColor[1];
      m_material.albedo[2] = diffuseColor[2];
    }

    if (boxedEmissiveColorIter != node.parameters.end())
    {
      VtValue vtValue = boxedEmissiveColorIter->second;;
      GfVec3f emissiveColor = vtValue.Get<GfVec3f>();

      m_material.emission[0] = emissiveColor[0];
      m_material.emission[1] = emissiveColor[1];
      m_material.emission[2] = emissiveColor[2];
    }
  }
}

HdDirtyBits HdGatlingMaterial::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams;
}

PXR_NAMESPACE_CLOSE_SCOPE
