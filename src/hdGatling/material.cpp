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
#include "utils.h"

#include <gtl/gi/Gi.h>

#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/types.h>

using namespace gtl;

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
  std::vector<GiPrimvarData> _GatherPrimvars(HdSceneDelegate* sceneDelegate, const SdfPath& id)
  {
    std::vector<GiPrimvarData> primvars;

    for (int i = 0; i < int(HdInterpolationCount); i++)
    {
      const auto& descriptors = sceneDelegate->GetPrimvarDescriptors(id, (HdInterpolation) i);

      for (const HdPrimvarDescriptor& descriptor : descriptors)
      {
        VtValue values = sceneDelegate->Get(id, descriptor.name);

        if (values.IsEmpty()) // UsdImagingDelegate expects the 'primvars:' attribute name for lookups
        {
          TfToken attributeName(SdfPath::JoinIdentifier(HdPrimvarsSchemaTokens->primvars, descriptor.name));
          values = sceneDelegate->Get(id, attributeName);
        }

        GiPrimvarInterpolation giInterpolation = HdGatlingGetGiPrimvarInterpolation(descriptor.interpolation);

        std::optional<GiPrimvarData> primvar = HdGatlingMakeGiPrimvarData(id, descriptor.name, std::move(values), giInterpolation);

        if (primvar)
        {
          primvars.push_back(std::move(*primvar));
        }
      }
    }

    return primvars;
  }
}

HdGatlingMaterial::HdGatlingMaterial(const SdfPath& id,
                                     GiScene* scene,
                                     const MaterialNetworkCompiler& materialNetworkCompiler)
  : HdMaterial(id)
  , _materialNetworkCompiler(materialNetworkCompiler)
  , _giScene(scene)
{
}

void HdGatlingMaterial::Finalize(HdRenderParam* renderParam)
{
  if (_giMaterial)
  {
    giDestroyMaterial(_giMaterial);
    _giMaterial = nullptr;
  }
}

HdDirtyBits HdGatlingMaterial::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams | DirtyBits::DirtyResource;
}

void HdGatlingMaterial::Sync(HdSceneDelegate* sceneDelegate,
                             HdRenderParam* renderParam,
                             HdDirtyBits* dirtyBits)
{
  TF_UNUSED(renderParam);

  bool pullMaterial = (*dirtyBits & (DirtyBits::DirtyParams | DirtyBits::DirtyResource));

  *dirtyBits = DirtyBits::Clean;

  if (!pullMaterial)
  {
    return;
  }

  if (_giMaterial)
  {
    giDestroyMaterial(_giMaterial);
    _giMaterial = nullptr;
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
  }

  bool hasSurface = network.terminals.count(HdMaterialTerminalTokens->surface) != 0;

  if (hasSurface)
  {
    _giMaterial = _materialNetworkCompiler.CompileNetwork(_giScene, id, network);
  }

  if (_giMaterial)
  {
    std::vector<GiPrimvarData> primvars = _GatherPrimvars(sceneDelegate, id);

    if (!primvars.empty())
    {
      giSetMaterialPrimvars(_giMaterial, primvars);
    }
  }
}

GiMaterial* HdGatlingMaterial::GetGiMaterial() const
{
  return _giMaterial;
}

PXR_NAMESPACE_CLOSE_SCOPE
