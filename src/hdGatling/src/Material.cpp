#include "Material.h"

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingMaterial::HdGatlingMaterial(const SdfPath& id,
                                     const MaterialNetworkTranslator& translator)
  : HdMaterial(id)
  , m_translator(translator)
{
}

HdGatlingMaterial::~HdGatlingMaterial()
{
  if (m_giMaterial)
  {
    giDestroyMaterial(m_giMaterial);
  }
}

HdDirtyBits HdGatlingMaterial::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams;
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
  const VtValue& resource = sceneDelegate->GetMaterialResource(id);

  if (!resource.IsHolding<HdMaterialNetworkMap>())
  {
    return;
  }

  const HdMaterialNetworkMap& networkMap = resource.UncheckedGet<HdMaterialNetworkMap>();
  bool isVolume = false;

  HdMaterialNetwork2 network = HdConvertToHdMaterialNetwork2(networkMap, &isVolume);
  if (isVolume)
  {
    TF_WARN("Volume %s unsupported", id.GetText());
    return;
  }

  m_giMaterial = m_translator.ParseNetwork(id, network);
}

const gi_material* HdGatlingMaterial::GetGiMaterial() const
{
  return m_giMaterial;
}

PXR_NAMESPACE_CLOSE_SCOPE
