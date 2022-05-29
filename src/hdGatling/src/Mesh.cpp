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

#include "Mesh.h"

#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/vertexAdjacency.h>
#include <pxr/imaging/hd/smoothNormals.h>
#include <pxr/imaging/hd/instancer.h>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingMesh::HdGatlingMesh(const SdfPath& id)
  : HdMesh(id)
  , m_prototypeTransform(1.0)
  , m_color(0.0, 0.0, 0.0)
  , m_hasColor(false)
{
}

HdGatlingMesh::~HdGatlingMesh()
{
}

void HdGatlingMesh::Sync(HdSceneDelegate* sceneDelegate,
                         HdRenderParam* renderParam,
                         HdDirtyBits* dirtyBits,
                         const TfToken& reprToken)
{
  TF_UNUSED(renderParam);
  TF_UNUSED(reprToken);

  HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();

  if ((*dirtyBits & HdChangeTracker::DirtyInstancer) |
      (*dirtyBits & HdChangeTracker::DirtyInstanceIndex))
  {
    HdDirtyBits dirtyBitsCopy = *dirtyBits;

    _UpdateInstancer(sceneDelegate, &dirtyBitsCopy);

    const SdfPath& instancerId = GetInstancerId();

    HdInstancer::_SyncInstancerAndParents(renderIndex, instancerId);
  }

  const SdfPath& id = GetId();

  if (*dirtyBits & HdChangeTracker::DirtyMaterialId)
  {
    const SdfPath& materialId = sceneDelegate->GetMaterialId(id);

    SetMaterialId(materialId);
  }

  if (*dirtyBits & HdChangeTracker::DirtyTransform)
  {
    m_prototypeTransform = sceneDelegate->GetTransform(id);
  }

  bool updateGeometry =
    (*dirtyBits & HdChangeTracker::DirtyPoints) |
    (*dirtyBits & HdChangeTracker::DirtyNormals) |
    (*dirtyBits & HdChangeTracker::DirtyTopology);

  *dirtyBits = HdChangeTracker::Clean;

  if (!updateGeometry)
  {
    return;
  }

  m_faces.clear();
  m_points.clear();
  m_normals.clear();

  _UpdateGeometry(sceneDelegate);
}

void HdGatlingMesh::_UpdateGeometry(HdSceneDelegate* sceneDelegate)
{
  const HdMeshTopology& topology = GetMeshTopology(sceneDelegate);
  const SdfPath& id = GetId();
  HdMeshUtil meshUtil(&topology, id);

  VtVec3iArray indices;
  VtIntArray primitiveParams;
  meshUtil.ComputeTriangleIndices(&indices, &primitiveParams);

  VtVec3fArray points;
  VtVec3fArray normals;
  bool indexedNormals;
  _PullPrimvars(sceneDelegate, points, normals, indexedNormals, m_color, m_hasColor);

  for (int i = 0; i < indices.size(); i++)
  {
    GfVec3i newFaceIndices(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    m_faces.push_back(newFaceIndices);

    const GfVec3i& faceIndices = indices[i];
    m_points.push_back(points[faceIndices[0]]);
    m_points.push_back(points[faceIndices[1]]);
    m_points.push_back(points[faceIndices[2]]);
    m_normals.push_back(normals[indexedNormals ? faceIndices[0] : newFaceIndices[0]]);
    m_normals.push_back(normals[indexedNormals ? faceIndices[1] : newFaceIndices[1]]);
    m_normals.push_back(normals[indexedNormals ? faceIndices[2] : newFaceIndices[2]]);
  }
}

bool HdGatlingMesh::_FindPrimvar(HdSceneDelegate* sceneDelegate,
                                 TfToken primvarName,
                                 HdInterpolation& interpolation) const
{
  HdInterpolation interpolations[] = {
    HdInterpolation::HdInterpolationVertex,
    HdInterpolation::HdInterpolationFaceVarying,
    HdInterpolation::HdInterpolationConstant,
    HdInterpolation::HdInterpolationUniform,
    HdInterpolation::HdInterpolationVarying,
    HdInterpolation::HdInterpolationInstance
  };

  for (HdInterpolation i : interpolations)
  {
    const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, i);

    for (const HdPrimvarDescriptor& primvar : primvarDescs)
    {
      if (primvar.name == primvarName)
      {
        interpolation = i;

        return true;
      }
    }
  }

  return false;
}

void HdGatlingMesh::_PullPrimvars(HdSceneDelegate* sceneDelegate,
                                  VtVec3fArray& points,
                                  VtVec3fArray& normals,
                                  bool& indexedNormals,
                                  GfVec3f& color,
                                  bool& hasColor) const
{
  const SdfPath& id = GetId();

  // Handle points.
  HdInterpolation pointInterpolation;
  bool foundPoints = _FindPrimvar(sceneDelegate,
                                  HdTokens->points,
                                  pointInterpolation);

  if (!foundPoints)
  {
    TF_RUNTIME_ERROR("Points primvar not found!");
    return;
  }
  else if (pointInterpolation != HdInterpolation::HdInterpolationVertex)
  {
    TF_RUNTIME_ERROR("Points primvar is not vertex-interpolated!");
    return;
  }

  VtValue boxedPoints = sceneDelegate->Get(id, HdTokens->points);
  points = boxedPoints.Get<VtVec3fArray>();

  // Handle color.
  HdInterpolation colorInterpolation;
  bool foundColor = _FindPrimvar(sceneDelegate,
                                 HdTokens->displayColor,
                                 colorInterpolation);

  if (foundColor && colorInterpolation == HdInterpolation::HdInterpolationConstant)
  {
    VtValue boxedColors = sceneDelegate->Get(id, HdTokens->displayColor);
    const VtVec3fArray& colors = boxedColors.Get<VtVec3fArray>();
    color = colors[0];
    hasColor = true;
  }

  // Handle normals.
  HdInterpolation normalInterpolation;
  bool foundNormals = _FindPrimvar(sceneDelegate,
                                   HdTokens->normals,
                                   normalInterpolation);

  if (foundNormals &&
      normalInterpolation == HdInterpolation::HdInterpolationVertex)
  {
    VtValue boxedNormals = sceneDelegate->Get(id, HdTokens->normals);
    normals = boxedNormals.Get<VtVec3fArray>();
    indexedNormals = true;
    return;
  }

  HdMeshTopology topology = GetMeshTopology(sceneDelegate);

  if (foundNormals &&
      normalInterpolation == HdInterpolation::HdInterpolationFaceVarying)
  {
    VtValue boxedFvNormals = sceneDelegate->Get(id, HdTokens->normals);
    const VtVec3fArray& fvNormals = boxedFvNormals.Get<VtVec3fArray>();

    HdMeshUtil meshUtil(&topology, id);
    VtValue boxedTriangulatedNormals;
    if (!meshUtil.ComputeTriangulatedFaceVaryingPrimvar(
        fvNormals.cdata(),
        fvNormals.size(),
        HdTypeFloatVec3,
        &boxedTriangulatedNormals))
    {
      TF_CODING_ERROR("Unable to triangulate face-varying normals of %s", id.GetText());
    }

    normals = boxedTriangulatedNormals.Get<VtVec3fArray>();
    indexedNormals = false;
    return;
  }

  Hd_VertexAdjacency adjacency;
  adjacency.BuildAdjacencyTable(&topology);
  normals = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, points.size(), points.cdata());
  indexedNormals = true;
}

const TfTokenVector BUILTIN_PRIMVAR_NAMES =
{
  HdTokens->points,
  HdTokens->normals
};

const TfTokenVector& HdGatlingMesh::GetBuiltinPrimvarNames() const
{
  return BUILTIN_PRIMVAR_NAMES;
}

const std::vector<GfVec3f>& HdGatlingMesh::GetPoints() const
{
  return m_points;
}

const std::vector<GfVec3f>& HdGatlingMesh::GetNormals() const
{
  return m_normals;
}

const std::vector<GfVec3i>& HdGatlingMesh::GetFaces() const
{
  return m_faces;
}

const GfMatrix4d& HdGatlingMesh::GetPrototypeTransform() const
{
  return m_prototypeTransform;
}

const GfVec3f& HdGatlingMesh::GetColor() const
{
  return m_color;
}

bool HdGatlingMesh::HasColor() const
{
  return m_hasColor;
}

HdDirtyBits HdGatlingMesh::GetInitialDirtyBitsMask() const
{
  return HdChangeTracker::DirtyPoints |
         HdChangeTracker::DirtyNormals |
         HdChangeTracker::DirtyTopology |
         HdChangeTracker::DirtyInstancer |
         HdChangeTracker::DirtyInstanceIndex |
         HdChangeTracker::DirtyTransform |
         HdChangeTracker::DirtyMaterialId;
}

HdDirtyBits HdGatlingMesh::_PropagateDirtyBits(HdDirtyBits bits) const
{
  return bits;
}

void HdGatlingMesh::_InitRepr(const TfToken& reprName,
                              HdDirtyBits *dirtyBits)
{
  TF_UNUSED(reprName);
  TF_UNUSED(dirtyBits);
}

PXR_NAMESPACE_CLOSE_SCOPE
