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

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingMesh final : public HdMesh
{
public:
  template<typename T>
  struct VertexAttr
  {
    VtArray<T> array;
    bool indexed;
  };

public:
  HdGatlingMesh(const SdfPath& id);

  ~HdGatlingMesh() override;

public:
  void Sync(HdSceneDelegate* delegate,
            HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits,
            const TfToken& reprToken) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

  const VtVec3iArray& GetFaces() const;
  const VtVec3fArray& GetPoints() const;
  const VertexAttr<GfVec3f>& GetNormals() const;
  const VertexAttr<GfVec2f>& GetTexCoords() const;

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
                    TfToken name,
                    HdInterpolation& interpolation) const;

  bool _ReadTriangulatedPrimvar(HdSceneDelegate* sceneDelegate,
                                VtIntArray primitiveParams,
                                TfToken name,
                                HdType type,
                                bool& sequentiallyIndexed,
                                VtValue& result) const;

  void _PullPrimvars(HdSceneDelegate* sceneDelegate,
                     VtIntArray primitiveParams,
                     GfVec3f& color,
                     bool& hasColor);

private:
  GfMatrix4d m_prototypeTransform;
  VtVec3iArray m_faces;
  VtVec3fArray m_points;
  VertexAttr<GfVec3f> m_normals;
  VertexAttr<GfVec2f> m_texCoords;
  GfVec3f m_color;
  bool m_hasColor;
};

PXR_NAMESPACE_CLOSE_SCOPE
