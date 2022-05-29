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

#include <gi.h>

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

  const TfTokenVector& GetBuiltinPrimvarNames() const override;

  const std::vector<GfVec3f>& GetPoints() const;

  const std::vector<GfVec3f>& GetNormals() const;

  const std::vector<GfVec3i>& GetFaces() const;

  const GfMatrix4d& GetPrototypeTransform() const;

  const GfVec3f& GetColor() const;

  bool HasColor() const;

protected:
  HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;

  void _InitRepr(const TfToken& reprName,
                 HdDirtyBits *dirtyBits) override;

private:
  void _UpdateGeometry(HdSceneDelegate* sceneDelegate);

  bool _FindPrimvar(HdSceneDelegate* sceneDelegate,
                    TfToken primvarName,
                    HdInterpolation& interpolation) const;

  void _PullPrimvars(HdSceneDelegate* sceneDelegate,
                     VtVec3fArray& points,
                     VtVec3fArray& normals,
                     bool& indexedNormals,
                     GfVec3f& color,
                     bool& hasColor) const;

private:
  GfMatrix4d m_prototypeTransform;
  std::vector<GfVec3f> m_points;
  std::vector<GfVec3f> m_normals;
  std::vector<GfVec3i> m_faces;
  GfVec3f m_color;
  bool m_hasColor;
};

PXR_NAMESPACE_CLOSE_SCOPE
