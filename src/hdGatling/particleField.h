//
// Copyright (C) 2026 Pablo Delgado Krämer
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

#include <pxr/imaging/hd/rprim.h>

namespace gtl
{
  struct GiMaterial;
  struct GiMesh;
  struct GiScene;
};

PXR_NAMESPACE_OPEN_SCOPE

using namespace gtl;

class HdGatlingParticleField : public HdRprim
{
public:
  /*struct GsMaterials
  {
    GiMaterial* sh0;
  };*/
public:
  HdGatlingParticleField(const SdfPath& id, GiScene* scene, GiMaterial* sh0Mat);

  ~HdGatlingParticleField();

  HdDirtyBits GetInitialDirtyBitsMask() const override;

  void Sync(HdSceneDelegate* sceneDelegate,
            HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits,
            TfToken const& reprToken) override;

  const TfTokenVector& GetBuiltinPrimvarNames() const override;

protected:
  void _InitRepr(TfToken const& reprToken, HdDirtyBits* dirtyBits) override;

  HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;

private:
  GiScene* _giScene;
  GiMaterial* _sh0Mat;
  GiMesh* _ellipseMesh = nullptr;
};

PXR_NAMESPACE_CLOSE_SCOPE
