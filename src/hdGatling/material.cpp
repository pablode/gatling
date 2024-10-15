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

#include "material.h"
#include "materialNetworkCompiler.h"

#include <gtl/gi/Gi.h>

using namespace gtl;

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingMaterial::HdGatlingMaterial(const SdfPath& id,
                                     const MaterialNetworkCompiler& materialNetworkCompiler)
  : HdMaterial(id)
  , _materialNetworkCompiler(materialNetworkCompiler)
{
}

HdGatlingMaterial::~HdGatlingMaterial()
{
  if (_giMaterial)
  {
    giDestroyMaterial(_giMaterial);
    _giMaterial = nullptr;
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
    if (_giMaterial)
    {
      giDestroyMaterial(_giMaterial);
      _giMaterial = nullptr;
    }
    return;
  }

  const HdMaterialNetworkMap& networkMap = resource.UncheckedGet<HdMaterialNetworkMap>();
  bool isVolume = false;

  HdMaterialNetwork2 network = HdConvertToHdMaterialNetwork2(networkMap, &isVolume);
  if (isVolume)
  {
    TF_WARN("Volume %s unsupported", id.GetText());

    if (_giMaterial)
    {
      giDestroyMaterial(_giMaterial);
      _giMaterial = nullptr;
    }
    return;
  }

  _giMaterial = _materialNetworkCompiler.CompileNetwork(id, network);
}

const GiMaterial* HdGatlingMaterial::GetGiMaterial() const
{
  return _giMaterial;
}

PXR_NAMESPACE_CLOSE_SCOPE
