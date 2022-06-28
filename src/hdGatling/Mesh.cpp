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
#include <pxr/imaging/hd/vtBufferSource.h>
#include <pxr/usd/usdUtils/pipeline.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  (st)
  (st0)
  (st_0)
  (st1)
  (st_1)
);

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

  m_faces = {};
  m_points = {};
  m_normals = {};
  m_texCoords = {};

  _UpdateGeometry(sceneDelegate);
}

void HdGatlingMesh::_UpdateGeometry(HdSceneDelegate* sceneDelegate)
{
  const HdMeshTopology& topology = GetMeshTopology(sceneDelegate);
  const SdfPath& id = GetId();
  HdMeshUtil meshUtil(&topology, id);

  VtIntArray primitiveParams;
  meshUtil.ComputeTriangleIndices(&m_faces, &primitiveParams);

  bool indexedNormals;
  _PullPrimvars(sceneDelegate, primitiveParams, m_color, m_hasColor);
}

bool HdGatlingMesh::_FindPrimvar(HdSceneDelegate* sceneDelegate,
                                 TfToken name,
                                 HdInterpolation& interpolation) const
{
  for (int i = 0; i < int(HdInterpolationCount); i++)
  {
    interpolation = (HdInterpolation) i;

    const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, interpolation);

    for (const HdPrimvarDescriptor& primvar : primvarDescs)
    {
      if (primvar.name == name)
      {
        return true;
      }
    }
  }

  return false;
}

template<typename T>
VtValue _ExpandBufferElements(const HdVtBufferSource& buffer, int elementExpansion)
{
  VtArray<T> result(buffer.GetNumElements() * elementExpansion);

  for (int i = 0; i < buffer.GetNumElements(); i++)
  {
    for (int j = 0; j < elementExpansion; j++)
    {
      result[i * elementExpansion + j] = ((T*) buffer.GetData())[i];
    }
  }

  return VtValue(std::move(result));
}

VtValue _ExpandBufferElements(const HdVtBufferSource& buffer, HdType type, int elementExpansion)
{
  if (type == HdTypeFloatVec3)
  {
    return _ExpandBufferElements<GfVec3f>(buffer, elementExpansion);
  }
  else if (type == HdTypeFloatVec2)
  {
    return _ExpandBufferElements<GfVec2f>(buffer, elementExpansion);
  }
  assert(false);
  return VtValue();
}

bool HdGatlingMesh::_ReadTriangulatedPrimvar(HdSceneDelegate* sceneDelegate,
                                             VtIntArray primitiveParams,
                                             TfToken name,
                                             HdType type,
                                             bool& isIndexed,
                                             VtValue& result) const
{
  HdInterpolation interpolation;
  if (!_FindPrimvar(sceneDelegate, name, interpolation))
  {
    return false;
  }

  const SdfPath& id = GetId();

  VtValue boxedValues = sceneDelegate->Get(id, name);
  HdVtBufferSource buffer(name, boxedValues);

  if (interpolation == HdInterpolationVertex)
  {
    result = boxedValues;
    isIndexed = true;
  }
  else if (interpolation == HdInterpolationConstant)
  {
    result = _ExpandBufferElements(buffer, type, primitiveParams.size());
    isIndexed = true;
  }
  else if (interpolation == HdInterpolationFaceVarying)
  {
    HdMeshTopology topology = GetMeshTopology(sceneDelegate);
    HdMeshUtil meshUtil(&topology, id);
    if (!meshUtil.ComputeTriangulatedFaceVaryingPrimvar(buffer.GetData(),
                                                        buffer.GetNumElements(),
                                                        type,
                                                        &result))
    {
      return false;
    }
    isIndexed = false;
  }
  else if (interpolation == HdInterpolationUniform)
  {
    result = _ExpandBufferElements(buffer, type, 3);
    uint8_t* dstPtr = (uint8_t*) HdGetValueData(result);
    uint8_t* srcPtr = (uint8_t*) HdGetValueData(boxedValues);
    size_t elementSize = HdDataSizeOfType(type);

    for (int faceIndex = 0; faceIndex < primitiveParams.size(); faceIndex++)
    {
      int oldFaceIndex = HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(primitiveParams[faceIndex]);

      memcpy(&dstPtr[(faceIndex * 3 + 0) * elementSize], &srcPtr[oldFaceIndex * elementSize], elementSize);
      memcpy(&dstPtr[(faceIndex * 3 + 1) * elementSize], &srcPtr[oldFaceIndex * elementSize], elementSize);
      memcpy(&dstPtr[(faceIndex * 3 + 2) * elementSize], &srcPtr[oldFaceIndex * elementSize], elementSize);
    }
    isIndexed = false;
  }
  else
  {
    return false;
  }

  return true;
}

void HdGatlingMesh::_PullPrimvars(HdSceneDelegate* sceneDelegate,
                                  VtIntArray primitiveParams,
                                  GfVec3f& color,
                                  bool& hasColor)
{
  const SdfPath& id = GetId();

  // Points: required per vertex.
  HdInterpolation pointInterpolation;
  bool foundPoints = _FindPrimvar(sceneDelegate, HdTokens->points, pointInterpolation);

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
  m_points = boxedPoints.Get<VtVec3fArray>();

  // Colors: only support constant interpolation because we can create a material for it.
  HdInterpolation colorInterpolation;
  bool foundColor = _FindPrimvar(sceneDelegate, HdTokens->displayColor, colorInterpolation);

  if (foundColor && colorInterpolation == HdInterpolation::HdInterpolationConstant)
  {
    VtValue boxedColors = sceneDelegate->Get(id, HdTokens->displayColor);
    const VtVec3fArray& colors = boxedColors.Get<VtVec3fArray>();
    color = colors[0];
    hasColor = true;
  }

  // Normals: calculate them from the topology if no primvar exists.
  VtValue boxedNormals;
  bool areNormalsIndexed;
  bool foundNormals = _ReadTriangulatedPrimvar(sceneDelegate,
                                               primitiveParams,
                                               HdTokens->normals,
                                               HdTypeFloatVec3,
                                               areNormalsIndexed,
                                               boxedNormals);

  if (foundNormals)
  {
    m_normals.array = boxedNormals.Get<VtVec3fArray>();
    m_normals.indexed = areNormalsIndexed;
  }
  else
  {
    HdMeshTopology topology = GetMeshTopology(sceneDelegate);

    Hd_VertexAdjacency adjacency;
    adjacency.BuildAdjacencyTable(&topology);
    m_normals.array = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, m_points.size(), m_points.cdata());
    m_normals.indexed = true;
  }

  // Tex Coords: since there is no standardization in respect to multiple sets, we have to guess.
  TfToken defaultUvPrimvarName = UsdUtilsGetPrimaryUVSetName();

  TfToken texcoordPrimvarNames[] = {
    defaultUvPrimvarName,
    _tokens->st,
    _tokens->st0,
    _tokens->st_0,
    _tokens->st1,
    _tokens->st_1
  };

  VtValue boxedTexCoords;
  for (const TfToken& name : texcoordPrimvarNames)
  {
    bool isIndexed;
    if (_ReadTriangulatedPrimvar(sceneDelegate,
                                 primitiveParams,
                                 name,
                                 HdTypeFloatVec2,
                                 isIndexed,
                                 boxedTexCoords))
    {
      m_texCoords.array = boxedTexCoords.Get<VtVec2fArray>();
      m_texCoords.indexed = isIndexed;
      break;
    }
  }
}

const VtVec3iArray& HdGatlingMesh::GetFaces() const
{
  return m_faces;
}

const VtVec3fArray& HdGatlingMesh::GetPoints() const
{
  return m_points;
}

const HdGatlingMesh::VertexAttr<GfVec3f>& HdGatlingMesh::GetNormals() const
{
  return m_normals;
}

const HdGatlingMesh::VertexAttr<GfVec2f>& HdGatlingMesh::GetTexCoords() const
{
  return m_texCoords;
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
