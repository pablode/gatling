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

#pragma once

#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/sceneDelegate.h>

namespace gtl
{
  struct GiMaterial;
  struct GiScene;
}

PXR_NAMESPACE_OPEN_SCOPE

class MaterialNetworkCompiler;

class HdGatlingMaterial final : public HdMaterial
{
public:
  HdGatlingMaterial(const SdfPath& id,
                    gtl::GiScene* scene,
                    const MaterialNetworkCompiler& materialNetworkCompiler);

  ~HdGatlingMaterial() override;

public:
  HdDirtyBits GetInitialDirtyBitsMask() const override;

  void Sync(HdSceneDelegate* sceneDelegate,
            HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits) override;

public:
  gtl::GiMaterial* GetGiMaterial() const;

private:
  const MaterialNetworkCompiler& _materialNetworkCompiler;
  gtl::GiMaterial* _giMaterial = nullptr;
  gtl::GiScene* _giScene = nullptr;
};

PXR_NAMESPACE_CLOSE_SCOPE
