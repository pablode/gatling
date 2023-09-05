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

#include "gi.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  (st)
  (st0)
  (st_0)
  (st1)
  (st_1)
  (UV0)
  (UV1)
  (tangents)
  (tangentSigns)
  (bitangentSigns)
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

  HdDirtyBits dirtyBitsCopy = *dirtyBits;

  HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();

  const SdfPath& id = GetId();

  if (*dirtyBits & HdChangeTracker::DirtyDoubleSided)
  {
    m_doubleSided = sceneDelegate->GetDoubleSided(id);
  }

  if ((*dirtyBits & HdChangeTracker::DirtyInstancer) |
      (*dirtyBits & HdChangeTracker::DirtyInstanceIndex))
  {

    _UpdateInstancer(sceneDelegate, &dirtyBitsCopy);

    const SdfPath& instancerId = GetInstancerId();

    HdInstancer::_SyncInstancerAndParents(renderIndex, instancerId);
  }

  if (*dirtyBits & HdChangeTracker::DirtyMaterialId)
  {
    const SdfPath& materialId = sceneDelegate->GetMaterialId(id);

    SetMaterialId(materialId);

    giInvalidateGeomCache(); // FIXME: remove this hack
  }

  if (*dirtyBits & HdChangeTracker::DirtyVisibility)
  {
    _UpdateVisibility(sceneDelegate, &dirtyBitsCopy);
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

  _PullPrimvars(sceneDelegate, primitiveParams, m_color, m_hasColor);
}

bool HdGatlingMesh::_FindPrimvarInterpolationByName(HdSceneDelegate* sceneDelegate,
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

TfToken HdGatlingMesh::_FindPrimvarByRole(HdSceneDelegate* sceneDelegate,
                                          TfToken role) const
{
  for (int i = 0; i < int(HdInterpolationCount); i++)
  {
    const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, (HdInterpolation) i);

    for (const HdPrimvarDescriptor& primvar : primvarDescs)
    {
      if (primvar.role == role)
      {
        return primvar.name;
      }
    }
  }

  return TfToken();
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
  if (type == HdTypeFloatVec4)
  {
    return _ExpandBufferElements<GfVec4f>(buffer, elementExpansion);
  }
  else if (type == HdTypeFloatVec3)
  {
    return _ExpandBufferElements<GfVec3f>(buffer, elementExpansion);
  }
  else if (type == HdTypeFloatVec2)
  {
    return _ExpandBufferElements<GfVec2f>(buffer, elementExpansion);
  }
  else if (type == HdTypeFloat)
  {
    return _ExpandBufferElements<float>(buffer, elementExpansion);
  }
  TF_VERIFY(false);
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
  if (!_FindPrimvarInterpolationByName(sceneDelegate, name, interpolation))
  {
    return false;
  }

  const SdfPath& id = GetId();

  VtValue boxedValues = sceneDelegate->Get(id, name);

  if ((type == HdTypeFloatVec4 && !boxedValues.IsHolding<VtVec4fArray>()) ||
      (type == HdTypeFloatVec3 && !boxedValues.IsHolding<VtVec3fArray>()) ||
      (type == HdTypeFloatVec2 && !boxedValues.IsHolding<VtVec2fArray>()) ||
      (type == HdTypeFloat     && !boxedValues.IsHolding<VtFloatArray>()))
  {
    return false;
  }

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
  else if (interpolation == HdInterpolationInstance)
  {
    TF_CODING_ERROR("primvar interpolation mode 'instance' not supported");
    return false;
  }
  else
  {
    TF_CODING_ERROR("primvar interpolation mode not handled");
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
  bool foundPoints = _FindPrimvarInterpolationByName(sceneDelegate, HdTokens->points, pointInterpolation);

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
  bool foundColor = _FindPrimvarInterpolationByName(sceneDelegate, HdTokens->displayColor, colorInterpolation);

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

  // Tex Coords: ideally should be read explicitly from primvars. But since this isn't implemented yet, we use
  //             heuristics to select a primvar likely containing tex coords. We start by checking well-known names.
  const TfToken texcoordPrimvarNameHints[] = {
     UsdUtilsGetPrimaryUVSetName(),
    _tokens->st,
    _tokens->st0,
    _tokens->st_0,
    _tokens->st1,
    _tokens->st_1,
    _tokens->UV0,
    _tokens->UV1
  };

  TfToken texcoordPrimvarName;
  for (const TfToken& name : texcoordPrimvarNameHints)
  {
    HdInterpolation unusedInterpolation;
    if (_FindPrimvarInterpolationByName(sceneDelegate, name, unusedInterpolation))
    {
      texcoordPrimvarName = name;
      break;
    }
  }

  // Otherwise, we select any primvar of a specific role.
  if (texcoordPrimvarName.IsEmpty())
  {
    texcoordPrimvarName = _FindPrimvarByRole(sceneDelegate, HdPrimvarRoleTokens->textureCoordinate);
  }

  if (!texcoordPrimvarName.IsEmpty())
  {
    VtValue boxedTexCoords;
    bool isIndexed;
    if (_ReadTriangulatedPrimvar(sceneDelegate,
                                 primitiveParams,
                                 texcoordPrimvarName,
                                 HdTypeFloatVec2,
                                 isIndexed,
                                 boxedTexCoords))
    {
      m_texCoords.array = boxedTexCoords.Get<VtVec2fArray>();
      m_texCoords.indexed = isIndexed;
    }
  }

  // Tangents & bitangents: either read combined vec4 array, or two separate primvars.
  VtValue boxedTangents;
  bool areTangentsIndexed;
  if (_ReadTriangulatedPrimvar(sceneDelegate,
                               primitiveParams,
                               _tokens->tangents,
                               HdTypeFloatVec4,
                               areTangentsIndexed,
                               boxedTangents))
  {
    m_tangents.indexed = areTangentsIndexed;
    m_bitangentSigns.indexed = areTangentsIndexed;

    VtVec3fArray& tangents = m_tangents.array;
    VtFloatArray& bitangentSigns = m_bitangentSigns.array;

    VtVec4fArray vec4Tangents = boxedTangents.Get<VtVec4fArray>();
    tangents.resize(vec4Tangents.size());
    bitangentSigns.resize(vec4Tangents.size());

    for (size_t i = 0; i < vec4Tangents.size(); i++)
    {
      tangents[i] = GfVec3f(vec4Tangents[i].data());
      bitangentSigns[i] = vec4Tangents[i][3];
    }
  }
  else if (_ReadTriangulatedPrimvar(sceneDelegate,
           primitiveParams,
           _tokens->tangents,
           HdTypeFloatVec3,
           areTangentsIndexed,
           boxedTangents))
  {
    m_tangents.indexed = areTangentsIndexed;
    m_tangents.array = boxedTangents.Get<VtVec3fArray>();

    const TfToken bitangentSignPrimvarNameHints[] = {
      _tokens->tangentSigns,   // <= guc 0.2
      _tokens->bitangentSigns  //  > guc 0.2
    };

    for (const TfToken& name : bitangentSignPrimvarNameHints)
    {
      VtValue boxedBitangentSigns;
      bool areBitangentSignsIndexed;
      if (_ReadTriangulatedPrimvar(sceneDelegate,
                                   primitiveParams,
                                   name,
                                   HdTypeFloat,
                                   areBitangentSignsIndexed,
                                   boxedBitangentSigns))
      {
        m_bitangentSigns.indexed = areBitangentSignsIndexed;
        m_bitangentSigns.array = boxedBitangentSigns.Get<VtFloatArray>();
      }
    }
  }
}

bool HdGatlingMesh::IsDoubleSided() const
{
  return m_doubleSided;
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

const HdGatlingMesh::VertexAttr<GfVec3f>& HdGatlingMesh::GetTangents() const
{
  return m_tangents;
}

const HdGatlingMesh::VertexAttr<float>& HdGatlingMesh::GetBitangentSigns() const
{
  return m_bitangentSigns;
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
         HdChangeTracker::DirtyMaterialId |
         HdChangeTracker::DirtyVisibility |
         HdChangeTracker::DirtyDoubleSided;
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
