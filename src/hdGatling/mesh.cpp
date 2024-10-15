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

#include "mesh.h"
#include "material.h"
#include "instancer.h"

#include <pxr/base/gf/matrix4f.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/vertexAdjacency.h>
#include <pxr/imaging/hd/smoothNormals.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/vtBufferSource.h>
#include <pxr/usd/usdUtils/pipeline.h>

#include <gtl/gi/Gi.h>

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
  (leftHanded)
);

namespace
{
  template<typename T>
  struct _VertexAttr
  {
    VtArray<T> array;
    bool indexed;
  };

  struct _VertexStreams
  {
    VtVec3iArray faces;
    VtVec3fArray points;
    _VertexAttr<GfVec3f> normals;
    _VertexAttr<GfVec2f> texCoords;
    _VertexAttr<GfVec3f> tangents;
    _VertexAttr<float> bitangentSigns;
  };

  GiVertex _MakeGiVertex(const GfMatrix4f& transform, const GfMatrix4f& normalMatrix, const GfVec3f& point, const GfVec3f& normal,
                         const GfVec2f& texCoords, const GfVec3f& tangent, float bitangentSign)
  {
    GfVec3f newPoint = transform.Transform(point);

    GfVec3f newNormal = normalMatrix.TransformDir(normal);
    newNormal.Normalize();

    GfVec3f newTangent = transform.Transform(tangent);
    newTangent.Normalize();

    GiVertex vertex;
    vertex.pos[0] = newPoint[0];
    vertex.pos[1] = newPoint[1];
    vertex.pos[2] = newPoint[2];
    vertex.norm[0] = newNormal[0];
    vertex.norm[1] = newNormal[1];
    vertex.norm[2] = newNormal[2];
    vertex.u = texCoords[0];
    vertex.v = 1.0f - texCoords[1];
    vertex.tangent[0] = newTangent[0];
    vertex.tangent[1] = newTangent[1];
    vertex.tangent[2] = newTangent[2];
    vertex.bitangentSign = bitangentSign;

    return vertex;
  }

  float _CalculateBitangentSign(const GfVec3f& n, const GfVec3f& t, const GfVec3f& b)
  {
    return (GfDot(GfCross(t, b), n) > 0.0f) ? 1.0f : -1.0f;
  }

  // Based on the algorithm proposed by Eric Lengyel in FGED 2 (Listing 7.4)
  // http://foundationsofgameenginedev.com/FGED2-sample.pdf
  void _CalculateTextureTangents(const VtVec3iArray& meshFaces,
                                 const VtVec3fArray& meshPoints,
                                 const _VertexAttr<GfVec3f>& meshNormals,
                                 const _VertexAttr<GfVec2f>& meshTexCoords,
                                 VtVec3fArray& meshTangents,
                                 VtFloatArray& meshBitangentSigns)
  {
    const float EPS = 0.0001f;
    size_t tangentCount = meshNormals.array.size();

    VtVec3fArray tangents(tangentCount, GfVec3f(0.0f));
    VtVec3fArray bitangents(tangentCount, GfVec3f(0.0f));
    VtVec3fArray normals(tangentCount, GfVec3f(0.0f));

    for (size_t i = 0; i < meshFaces.size(); i++)
    {
      const auto& f = meshFaces[i];

      const auto& p0 = meshPoints[f[0]];
      const auto& p1 = meshPoints[f[1]];
      const auto& p2 = meshPoints[f[2]];
      const auto& n0 = meshNormals.array[meshNormals.indexed ? f[0] : (i * 3 + 0)];
      const auto& n1 = meshNormals.array[meshNormals.indexed ? f[1] : (i * 3 + 1)];
      const auto& n2 = meshNormals.array[meshNormals.indexed ? f[2] : (i * 3 + 2)];
      const auto& t0 = meshTexCoords.array[meshTexCoords.indexed ? f[0] : (i * 3 + 0)];
      const auto& t1 = meshTexCoords.array[meshTexCoords.indexed ? f[1] : (i * 3 + 1)];
      const auto& t2 = meshTexCoords.array[meshTexCoords.indexed ? f[2] : (i * 3 + 2)];

      GfVec3f e1 = p1 - p0, e2 = p2 - p0;
      float x1 = t1[0] - t0[0], x2 = t2[0] - t0[0];
      float y1 = t1[1] - t0[1], y2 = t2[1] - t0[1];

      GfVec3f t, b;
      float denom = x1 * y2 - x2 * y1;

      // The original algorithm does not handle this special case, causing NaNs!
      if (fabsf(denom) > EPS)
      {
        float r = (1.0f / denom);

        t = (e1 * y2 - e2 * y1) * r;
        b = (e2 * x1 - e1 * x2) * r;
      }
      else
      {
        // Fall back to default UV direction
        t = GfVec3f::YAxis();
        b = GfVec3f::XAxis();
      }

      size_t outIndex0 = meshNormals.indexed ? f[0] : (i * 3 + 0);
      size_t outIndex1 = meshNormals.indexed ? f[1] : (i * 3 + 1);
      size_t outIndex2 = meshNormals.indexed ? f[2] : (i * 3 + 2);

      // Assets can author out-of-range indices (f.i. Intel's Sponza scene). Skip those.
      if (outIndex0 >= tangentCount || outIndex1 >= tangentCount || outIndex2 >= tangentCount)
      {
        TF_WARN("invalid primvar index; skipping");
        continue;
      }

      normals[outIndex0] += n0;
      normals[outIndex1] += n1;
      normals[outIndex2] += n2;
      tangents[outIndex0] += t;
      tangents[outIndex1] += t;
      tangents[outIndex2] += t;
      bitangents[outIndex0] += b;
      bitangents[outIndex1] += b;
      bitangents[outIndex2] += b;
    }

    meshTangents.resize(tangentCount);
    meshBitangentSigns.resize(tangentCount);

    for (size_t i = 0; i < tangentCount; i++)
    {
      const GfVec3f& n = meshNormals.array[i].GetNormalized();

      // Robust special-case handling based on the logic from DirectXMesh:
      // https://github.com/microsoft/DirectXMesh/blob/5647700332a2a2504000529902ac3164c058d616/DirectXMesh/DirectXMeshTangentFrame.cpp#L126-L162

      GfVec3f t = tangents[i];
      t = (t - n * GfDot(n, t)); // Gram-Schmidt re-orthogonalization

      GfVec3f b = bitangents[i];
      b = (b - n * GfDot(n, b)) - (t * GfDot(t, b));

      float tLen = t.GetLength();
      float bLen = b.GetLength();

      if (tLen > 0.0f)
      {
        t = t.GetNormalized();
      }
      if (bLen > 0.0f)
      {
        b = b.GetNormalized();
      }

      if (tLen <= EPS || bLen <= EPS)
      {
        if (tLen > 0.5f)
        {
          b = GfCross(n, t);
        }
        else if (bLen > 0.5f)
        {
          t = GfCross(b, n);
        }
        else
        {
          float d0 = abs(n[0]);
          float d1 = abs(n[1]);
          float d2 = abs(n[2]);

          GfVec3f axis;
          if (d0 < d1)
          {
            axis = (d0 < d2) ? GfVec3f::XAxis() : GfVec3f::ZAxis();
          }
          else if (d1 < d2)
          {
            axis = GfVec3f::YAxis();
          }
          else
          {
            axis = GfVec3f::ZAxis();
          }

          t = GfCross(n, axis);
          b = GfCross(n, t);
        }
      }

      meshTangents[i] = t;
      meshBitangentSigns[i] = _CalculateBitangentSign(n, t, b);
    }
  }

  // Duff et al. 2017. Building an Orthonormal Basis, Revisited. JCGT.
  // Licensed under CC BY-ND 3.0: https://creativecommons.org/licenses/by-nd/3.0/
  void _DuffOrthonormalBasis(const GfVec3f& n, GfVec3f& tangent, GfVec3f& bitangent)
  {
    float nSign = (n[2] >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (nSign + n[2]);
    float b = n[0] * n[1] * a;
    tangent = GfVec3f(1.0f + nSign * n[0] * n[0] * a, nSign * b, -nSign * n[0]);
    bitangent = GfVec3f(b, nSign + n[1] * n[1] * a, -n[1]);
  }

  void _CalculateFallbackTangents(const VtVec3iArray& meshFaces,
                                  const _VertexAttr<GfVec3f>& meshNormals,
                                  VtVec3fArray& meshTangents,
                                  VtFloatArray& meshBitangentSigns)
  {
    size_t normalCount = meshNormals.array.size();

    meshTangents.resize(normalCount);
    meshBitangentSigns.resize(normalCount);

    for (size_t i = 0; i < normalCount; i++)
    {
      const GfVec3f normal = meshNormals.array[i];

      GfVec3f tangent, bitangent;
      _DuffOrthonormalBasis(normal, tangent, bitangent);

      meshTangents[i] = tangent;
      meshBitangentSigns[i] = _CalculateBitangentSign(normal, tangent, bitangent);
    }
  }

  void _CalculateTangents(const VtVec3iArray& meshFaces,
                          const VtVec3fArray& meshPoints,
                          const _VertexAttr<GfVec3f>& meshNormals,
                          const _VertexAttr<GfVec2f>& meshTexCoords,
                          _VertexAttr<GfVec3f>& meshTangents,
                          _VertexAttr<float>& meshBitangentSigns)
  {
    bool hasTexCoords = meshTexCoords.array.size() > 0;

    if (hasTexCoords)
    {
      _CalculateTextureTangents(meshFaces, meshPoints, meshNormals, meshTexCoords, meshTangents.array, meshBitangentSigns.array);
    }
    else
    {
      _CalculateFallbackTangents(meshFaces, meshNormals, meshTangents.array, meshBitangentSigns.array);
    }

    meshTangents.indexed = meshNormals.indexed;
    meshBitangentSigns.indexed = meshNormals.indexed;
  }

  void _BakeMeshGeometry(_VertexStreams& s,
                         const GfMatrix4f& transform,
                         std::vector<GiFace>& faces,
                         std::vector<GiVertex>& vertices)
  {
    GfMatrix4f normalMatrix(transform.GetInverse().GetTranspose());

    bool hasTexCoords = s.texCoords.array.size() > 0;
    bool calcTangents = s.tangents.array.empty();
    bool calcBitangentSigns = s.bitangentSigns.array.empty();

    if (!calcTangents && calcBitangentSigns)
    {
#if 0
      // If no bitangent signs have been found, chances are high that none have been authored in the first place.
      // Handedness may then be assumed to be positive, although force calculating the tangents could yield better results.
      calcTangents = true;
#else
      TF_WARN("tangents have been provided without handedness; assuming positive");
      size_t signCount = std::max(s.normals.array.size(), s.tangents.array.size());
      s.bitangentSigns.array = VtFloatArray(signCount, 1.0f);
      s.bitangentSigns.indexed = s.normals.indexed && s.tangents.indexed;
#endif
    }
    if (calcTangents)
    {
      _CalculateTangents(s.faces, s.points, s.normals, s.texCoords, s.tangents, s.bitangentSigns);
    }

    bool isAnyPrimvarNotIndexed = !s.normals.indexed || !s.texCoords.indexed || !s.tangents.indexed;
    uint32_t vertexOffset = vertices.size();

    for (size_t i = 0; i < s.faces.size(); i++)
    {
      const GfVec3i& vertexIndices = s.faces[i];

      GiFace face;
      face.v_i[0] = vertexOffset + (isAnyPrimvarNotIndexed ? (i * 3 + 0) : vertexIndices[0]);
      face.v_i[1] = vertexOffset + (isAnyPrimvarNotIndexed ? (i * 3 + 1) : vertexIndices[1]);
      face.v_i[2] = vertexOffset + (isAnyPrimvarNotIndexed ? (i * 3 + 2) : vertexIndices[2]);

      // We always need three unique vertices per face.
      if (isAnyPrimvarNotIndexed)
      {
        for (size_t j = 0; j < 3; j++)
        {
          const GfVec3f& point = s.points[vertexIndices[j]];
          const GfVec3f& normal = s.normals.array[s.normals.indexed ? vertexIndices[j] : (i * 3 + j)];
          GfVec2f texCoords = hasTexCoords ? s.texCoords.array[s.texCoords.indexed ? vertexIndices[j] : (i * 3 + j)] : GfVec2f();

          GfVec3f tangent = s.tangents.array[s.tangents.indexed ? vertexIndices[j] : (i * 3 + j)];
          float bitangentSign = s.bitangentSigns.array[s.bitangentSigns.indexed ? vertexIndices[j] : (i * 3 + j)];

          GiVertex vertex = _MakeGiVertex(transform, normalMatrix, point, normal, texCoords, tangent, bitangentSign);
          vertices.push_back(vertex);
        }
      }

      faces.push_back(face);
    }

    // Early-out if the vertices are not indexed.
    if (isAnyPrimvarNotIndexed)
    {
      return;
    }

    for (size_t j = 0; j < s.points.size(); j++)
    {
      const GfVec3f& point = s.points[j];
      const GfVec3f& normal = s.normals.array[j];
      GfVec2f texCoords = hasTexCoords ? s.texCoords.array[j] : GfVec2f();

      GfVec3f tangent = s.tangents.array[j];
      float bitangentSign = s.bitangentSigns.array[j];

      GiVertex vertex = _MakeGiVertex(transform, normalMatrix, point, normal, texCoords, tangent, bitangentSign);
      vertices.push_back(vertex);
    }
  }

}

HdGatlingMesh::HdGatlingMesh(const SdfPath& id,
                             GiScene* scene,
                             const GiMaterial* defaultMaterial)
  : HdMesh(id)
  , _color(0.0, 0.0, 0.0)
  , _hasColor(false)
  , _scene(scene)
  , _defaultMaterial(defaultMaterial)
{
}

HdGatlingMesh::~HdGatlingMesh()
{
  if (_giMesh)
  {
    giDestroyMesh(_giMesh);
  }
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

  if (*dirtyBits & HdChangeTracker::DirtyVisibility)
  {
    _UpdateVisibility(sceneDelegate, &dirtyBitsCopy);
  }

  bool updateGeometry =
    (*dirtyBits & HdChangeTracker::DirtyPoints) |
    (*dirtyBits & HdChangeTracker::DirtyNormals) |
    (*dirtyBits & HdChangeTracker::DirtyTopology);

  if (updateGeometry)
  {
    if (_giMesh)
    {
      giDestroyMesh(_giMesh);
      _giMesh = nullptr;
    }

    _CreateGiMesh(sceneDelegate);

    (*dirtyBits) |= HdChangeTracker::DirtyMaterialId; // force material assignment
  }

  if (_giMesh &&
      ((*dirtyBits & HdChangeTracker::DirtyInstancer) ||
      (*dirtyBits & HdChangeTracker::DirtyInstanceIndex)))
  {
    _UpdateInstancer(sceneDelegate, &dirtyBitsCopy);

    const SdfPath& instancerId = GetInstancerId();

    HdInstancer::_SyncInstancerAndParents(renderIndex, instancerId);

    VtMatrix4fArray transforms;
    if (instancerId.IsEmpty())
    {
      transforms.resize(1);
      transforms[0] = GfMatrix4f(1.0);
    }
    else
    {
      HdInstancer* boxedInstancer = renderIndex.GetInstancer(instancerId);
      HdGatlingInstancer* instancer = static_cast<HdGatlingInstancer*>(boxedInstancer);

      transforms = instancer->ComputeInstanceTransforms(id);
    }

    giSetMeshInstanceTransforms(_giMesh, uint32_t(transforms.size()),
                                (const float(*)[4][4]) transforms[0].data());
  }

  if (_giMesh && (*dirtyBits & HdChangeTracker::DirtyTransform))
  {
    GfMatrix4d t = sceneDelegate->GetTransform(id);

    float transform[3][4] = {
      { (float) t[0][0], (float) t[1][0], (float) t[2][0], (float) t[3][0] },
      { (float) t[0][1], (float) t[1][1], (float) t[2][1], (float) t[3][1] },
      { (float) t[0][2], (float) t[1][2], (float) t[2][2], (float) t[3][2] }
    };

    giSetMeshTransform(_giMesh, transform);
  }

  if (_giMesh && (*dirtyBits & HdChangeTracker::DirtyMaterialId))
  {
    const SdfPath& materialId = sceneDelegate->GetMaterialId(id);

    SetMaterialId(materialId);

    // Because Hydra syncs Rprims last, it is guaranteed that the material has been processed
    auto* materialPrim = static_cast<HdGatlingMaterial*>(renderIndex.GetSprim(HdPrimTypeTokens->material, materialId));

    const GiMaterial* giMat = materialPrim ? materialPrim->GetGiMaterial() : nullptr;

    if (!giMat)
    {
      giMat = _defaultMaterial;
    }

    giSetMeshMaterial(_giMesh, giMat);
  }

  *dirtyBits = HdChangeTracker::Clean;
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
VtValue _ExpandBufferElements(const HdVtBufferSource& buffer, size_t elementExpansion)
{
  VtArray<T> result(buffer.GetNumElements() * elementExpansion);

  for (size_t i = 0; i < buffer.GetNumElements(); i++)
  {
    for (size_t j = 0; j < elementExpansion; j++)
    {
      result[i * elementExpansion + j] = ((T*) buffer.GetData())[i];
    }
  }

  return VtValue(std::move(result));
}

VtValue _ExpandBufferElements(const HdVtBufferSource& buffer, HdType type, size_t elementExpansion)
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

  if (interpolation == HdInterpolationVertex ||
      // Varying is equivalent to Vertex for non-subdivided polygonal surfaces (and we don't support subdivision):
      // https://github.com/usd-wg/assets/tree/907d5f17bbe933fc14441a3f3ab69a5bd8abe32a/docs/PrimvarInterpolation#vertex
      interpolation == HdInterpolationVarying)
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

    for (size_t faceIndex = 0; faceIndex < primitiveParams.size(); faceIndex++)
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
    TF_CODING_ERROR("Primvar interpolation mode 'instance' not supported (%s)", id.GetText());
    return false;
  }
  else
  {
    TF_CODING_ERROR("Primvar interpolation mode not handled");
    return false;
  }

  return true;
}

void HdGatlingMesh::_CreateGiMesh(HdSceneDelegate* sceneDelegate)
{
  const SdfPath& id = GetId();

  _VertexStreams s;

  // Faces
  const HdMeshTopology& topology = GetMeshTopology(sceneDelegate);
  HdMeshUtil meshUtil(&topology, id);

  VtIntArray primitiveParams;
  meshUtil.ComputeTriangleIndices(&s.faces, &primitiveParams);

  // Points: required per vertex.
  VtValue boxedPoints = GetPoints(sceneDelegate);

  if (boxedPoints.IsEmpty())
  {
    TF_RUNTIME_ERROR("Points primvar not found (%s)", id.GetText());
    return;
  }

  s.points = boxedPoints.Get<VtVec3fArray>();

  // Colors: only support constant interpolation because we can create a material for it.
  HdInterpolation colorInterpolation;
  bool foundColor = _FindPrimvarInterpolationByName(sceneDelegate, HdTokens->displayColor, colorInterpolation);

  if (foundColor && colorInterpolation == HdInterpolation::HdInterpolationConstant)
  {
    VtValue boxedColors = sceneDelegate->Get(id, HdTokens->displayColor);
    const VtVec3fArray& colors = boxedColors.Get<VtVec3fArray>();
    _color = colors[0];
    _hasColor = true;
  }
  else
  {
    _hasColor = false;
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
    s.normals.array = boxedNormals.Get<VtVec3fArray>();
    s.normals.indexed = areNormalsIndexed;
  }
  else
  {
    Hd_VertexAdjacency adjacency;
    adjacency.BuildAdjacencyTable(&topology);
    s.normals.array = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, s.points.size(), s.points.cdata());
    s.normals.indexed = true;
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
      s.texCoords.array = boxedTexCoords.Get<VtVec2fArray>();
      s.texCoords.indexed = isIndexed;
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
    s.tangents.indexed = areTangentsIndexed;
    s.bitangentSigns.indexed = areTangentsIndexed;

    VtVec3fArray& tangents = s.tangents.array;
    VtFloatArray& bitangentSigns = s.bitangentSigns.array;

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
    s.tangents.indexed = areTangentsIndexed;
    s.tangents.array = boxedTangents.Get<VtVec3fArray>();

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
        s.bitangentSigns.indexed = areBitangentSignsIndexed;
        s.bitangentSigns.array = boxedBitangentSigns.Get<VtFloatArray>();
      }
    }
  }

  std::vector<GiFace> faces;
  std::vector<GiVertex> vertices;
  _BakeMeshGeometry(s, GfMatrix4f(1.0f), faces, vertices);

  TfToken orientation = topology.GetOrientation();
  bool isLeftHanded = (orientation == _tokens->leftHanded);

  GiMeshDesc desc = {
    .faceCount = (uint32_t) faces.size(),
    .faces = faces.data(),
    .id = GetPrimId(),
    .isLeftHanded = isLeftHanded,
    .vertexCount = (uint32_t) vertices.size(),
    .vertices = vertices.data()
  };
  _giMesh = giCreateMesh(_scene, desc);
}

GiMesh* HdGatlingMesh::GetGiMesh() const
{
  return _giMesh;
}

const GfVec3f& HdGatlingMesh::GetColor() const
{
  return _color;
}

bool HdGatlingMesh::HasColor() const
{
  return _hasColor;
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
