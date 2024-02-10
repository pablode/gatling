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

#include <pxr/imaging/hd/mesh.h>
#include <pxr/base/gf/vec2f.h>

namespace gtl
{
  struct GiMesh;
}

using namespace gtl;

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingMesh final : public HdMesh
{
public:
  HdGatlingMesh(const SdfPath& id);

  ~HdGatlingMesh() override;

public:
  void Sync(HdSceneDelegate* delegate,
            HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits,
            const TfToken& reprToken) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

  const GiMesh* GetGiMesh() const;

  const GfMatrix4d& GetPrototypeTransform() const;

  const GfVec3f& GetColor() const;

  bool HasColor() const;

protected:
  HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;

  void _InitRepr(const TfToken& reprName,
                 HdDirtyBits *dirtyBits) override;

private:
  bool _FindPrimvarInterpolationByName(HdSceneDelegate* sceneDelegate,
                                       TfToken name,
                                       HdInterpolation& interpolation) const;

  TfToken _FindPrimvarByRole(HdSceneDelegate* sceneDelegate,
                             TfToken role) const;

  bool _ReadTriangulatedPrimvar(HdSceneDelegate* sceneDelegate,
                                VtIntArray primitiveParams,
                                TfToken name,
                                HdType type,
                                bool& sequentiallyIndexed,
                                VtValue& result) const;

  void _CreateGiMesh(HdSceneDelegate* sceneDelegate);

private:
  GiMesh* _giMesh = nullptr;
  GfMatrix4d _prototypeTransform;
  GfVec3f _color;
  bool _hasColor = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
