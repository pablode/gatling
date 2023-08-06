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

#include "Material.h"

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingMaterial::HdGatlingMaterial(const SdfPath& id)
  : HdMaterial(id)
{
}

HdGatlingMaterial::~HdGatlingMaterial()
{
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
    m_network.reset();
    return;
  }

  const HdMaterialNetworkMap& networkMap = resource.UncheckedGet<HdMaterialNetworkMap>();
  bool isVolume = false;

  HdMaterialNetwork2 network = HdConvertToHdMaterialNetwork2(networkMap, &isVolume);
  if (isVolume)
  {
    TF_WARN("Volume %s unsupported", id.GetText());
    m_network.reset();
    return;
  }

  m_network = std::make_unique<HdMaterialNetwork2>();
  *m_network = network;

  giInvalidateShaderCache(); // FIXME: track dirty state in RenderParam
}

const HdMaterialNetwork2* HdGatlingMaterial::GetNetwork() const
{
  return m_network.get();
}

PXR_NAMESPACE_CLOSE_SCOPE
