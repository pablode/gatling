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
#include "utils.h"

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
  struct _VertexStreams
  {
    VtVec3iArray faces;
    VtVec3fArray points;
    VtVec3fArray normals;
    VtVec2fArray texCoords;
    VtVec3fArray tangents;
    VtFloatArray bitangentSigns;
  };

  GiVertex _MakeGiVertex(const GfVec3f& point, const GfVec3f& normal, const GfVec2f& texCoords, const GfVec3f& tangent, float bitangentSign)
  {
    GiVertex vertex;
    vertex.pos[0] = point[0];
    vertex.pos[1] = point[1];
    vertex.pos[2] = point[2];
    vertex.norm[0] = normal[0];
    vertex.norm[1] = normal[1];
    vertex.norm[2] = normal[2];
    vertex.u = texCoords[0];
    vertex.v = texCoords[1];
    vertex.tangent[0] = tangent[0];
    vertex.tangent[1] = tangent[1];
    vertex.tangent[2] = tangent[2];
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
                                 const VtVec3fArray& meshNormals,
                                 const VtVec2fArray& meshTexCoords,
                                 VtVec3fArray& meshTangents,
                                 VtFloatArray& meshBitangentSigns)
  {
    const float EPS = 0.0001f;
    size_t tangentCount = meshNormals.size();
    TF_AXIOM(tangentCount == meshPoints.size());

    VtVec3fArray tangents(tangentCount, GfVec3f(0.0f));
    VtVec3fArray bitangents(tangentCount, GfVec3f(0.0f));
    VtVec3fArray normals(tangentCount, GfVec3f(0.0f));

    for (size_t i = 0; i < meshFaces.size(); i++)
    {
      const auto& f = meshFaces[i];

      const auto& p0 = meshPoints[f[0]];
      const auto& p1 = meshPoints[f[1]];
      const auto& p2 = meshPoints[f[2]];
      const auto& n0 = meshNormals[f[0]];
      const auto& n1 = meshNormals[f[1]];
      const auto& n2 = meshNormals[f[2]];
      const auto& t0 = meshTexCoords[f[0]];
      const auto& t1 = meshTexCoords[f[1]];
      const auto& t2 = meshTexCoords[f[2]];

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

      size_t outIndex0 = f[0];
      size_t outIndex1 = f[1];
      size_t outIndex2 = f[2];

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
      const GfVec3f& n = meshNormals[i].GetNormalized();

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

  void _CalculateFallbackTangents(const VtVec3fArray& meshNormals,
                                  VtVec3fArray& meshTangents,
                                  VtFloatArray& meshBitangentSigns)
  {
    size_t normalCount = meshNormals.size();

    meshTangents.resize(normalCount);
    meshBitangentSigns.resize(normalCount);

    for (size_t i = 0; i < normalCount; i++)
    {
      const GfVec3f normal = meshNormals[i];

      GfVec3f tangent, bitangent;
      _DuffOrthonormalBasis(normal, tangent, bitangent);

      meshTangents[i] = tangent;
      meshBitangentSigns[i] = _CalculateBitangentSign(normal, tangent, bitangent);
    }
  }

  void _CalculateTangents(const VtVec3iArray& meshFaces,
                          const VtVec3fArray& meshPoints,
                          const VtVec3fArray& meshNormals,
                          const VtVec2fArray& meshTexCoords,
                          VtVec3fArray& meshTangents,
                          VtFloatArray& meshBitangentSigns)
  {
    bool hasTexCoords = meshTexCoords.size() > 0;

    if (hasTexCoords)
    {
      _CalculateTextureTangents(meshFaces, meshPoints, meshNormals, meshTexCoords, meshTangents, meshBitangentSigns);
    }
    else
    {
      _CalculateFallbackTangents(meshNormals, meshTangents, meshBitangentSigns);
    }
  }

  void _BakeMeshGeometry(_VertexStreams& s,
                         std::vector<GiFace>& faces,
                         std::vector<GiVertex>& vertices)
  {
    bool hasTexCoords = s.texCoords.size() > 0;
    bool calcTangents = s.tangents.empty();
    bool calcBitangentSigns = s.bitangentSigns.empty();

    if (!calcTangents && calcBitangentSigns)
    {
#if 0
      // If no bitangent signs have been found, chances are high that none have been authored in the first place.
      // Handedness may then be assumed to be positive, although force calculating the tangents could yield better results.
      calcTangents = true;
#else
      TF_WARN("tangents have been provided without handedness; assuming positive");
      size_t signCount = std::max(s.normals.size(), s.tangents.size());
      s.bitangentSigns = VtFloatArray(signCount, 1.0f);
#endif
    }
    if (calcTangents)
    {
      _CalculateTangents(s.faces, s.points, s.normals, s.texCoords, s.tangents, s.bitangentSigns);
    }

    for (size_t i = 0; i < s.faces.size(); i++)
    {
      const GfVec3i& vertexIndices = s.faces[i];

      GiFace face;
      face.v_i[0] = vertexIndices[0];
      face.v_i[1] = vertexIndices[1];
      face.v_i[2] = vertexIndices[2];

      faces.push_back(face);
    }

    for (size_t j = 0; j < s.points.size(); j++)
    {
      const GfVec3f& point = s.points[j];
      const GfVec3f& normal = s.normals[j];
      GfVec2f texCoords = hasTexCoords ? s.texCoords[j] : GfVec2f();

      GfVec3f tangent = s.tangents[j];
      float bitangentSign = s.bitangentSigns[j];

      GiVertex vertex = _MakeGiVertex(point, normal, texCoords, tangent, bitangentSign);
      vertices.push_back(vertex);
    }
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
    if (type == HdTypeFloatVec4)      return _ExpandBufferElements<GfVec4f>(buffer, elementExpansion);
    else if (type == HdTypeFloatVec3) return _ExpandBufferElements<GfVec3f>(buffer, elementExpansion);
    else if (type == HdTypeFloatVec2) return _ExpandBufferElements<GfVec2f>(buffer, elementExpansion);
    else if (type == HdTypeFloat)     return _ExpandBufferElements<float>(buffer, elementExpansion);
    else if (type == HdTypeInt32Vec4) return _ExpandBufferElements<GfVec4i>(buffer, elementExpansion);
    else if (type == HdTypeInt32Vec3) return _ExpandBufferElements<GfVec3i>(buffer, elementExpansion);
    else if (type == HdTypeInt32Vec2) return _ExpandBufferElements<GfVec2i>(buffer, elementExpansion);
    else if (type == HdTypeInt32)     return _ExpandBufferElements<int32_t>(buffer, elementExpansion);

    TF_AXIOM(false);
    return VtValue();
  }

  template<typename T>
  VtValue _DeindexBufferElements(const HdVtBufferSource& buffer, const VtVec3iArray& faces)
  {
    VtArray<T> result(faces.size() * 3);

    for (size_t i = 0; i < faces.size(); i++)
    {
      for (size_t j = 0; j < 3; j++)
      {
        const GfVec3i& f = faces[i];

        uint32_t srcIdx = f[j];
        uint32_t dstIdx = i * 3 + j;

        result[dstIdx] = ((T*) buffer.GetData())[srcIdx];
      }
    }

    return VtValue(std::move(result));
  }

  VtValue _DeindexBufferElements(HdType type, const HdVtBufferSource& buffer, const VtVec3iArray& faces)
  {
    if (type == HdTypeFloatVec4)      return _DeindexBufferElements<GfVec4f>(buffer, faces);
    else if (type == HdTypeFloatVec3) return _DeindexBufferElements<GfVec3f>(buffer, faces);
    else if (type == HdTypeFloatVec2) return _DeindexBufferElements<GfVec2f>(buffer, faces);
    else if (type == HdTypeFloat)     return _DeindexBufferElements<float>(buffer, faces);
    else if (type == HdTypeInt32Vec4) return _DeindexBufferElements<GfVec4i>(buffer, faces);
    else if (type == HdTypeInt32Vec3) return _DeindexBufferElements<GfVec3i>(buffer, faces);
    else if (type == HdTypeInt32Vec2) return _DeindexBufferElements<GfVec2i>(buffer, faces);
    else if (type == HdTypeInt32)     return _DeindexBufferElements<int32_t>(buffer, faces);

    TF_AXIOM(false);
    return VtValue();
  }

  VtValue _CreateSizedArray(HdType type, uint32_t elementCount)
  {
    if (type == HdTypeFloatVec4)      return VtValue(VtVec4fArray(elementCount));
    else if (type == HdTypeFloatVec3) return VtValue(VtVec3fArray(elementCount));
    else if (type == HdTypeFloatVec2) return VtValue(VtVec2fArray(elementCount));
    else if (type == HdTypeFloat)     return VtValue(VtFloatArray(elementCount));
    else if (type == HdTypeInt32Vec4) return VtValue(VtVec4iArray(elementCount));
    else if (type == HdTypeInt32Vec3) return VtValue(VtVec3iArray(elementCount));
    else if (type == HdTypeInt32Vec2) return VtValue(VtVec2iArray(elementCount));
    else if (type == HdTypeInt32)     return VtValue(VtIntArray(elementCount));

    TF_AXIOM(false);
    return VtValue();
  }

  const static TfToken _texcoordPrimvarNameHints[] = {
    UsdUtilsGetPrimaryUVSetName(),
    _tokens->st,
    _tokens->st0,
    _tokens->st_0,
    _tokens->st1,
    _tokens->st_1,
    _tokens->UV0,
    _tokens->UV1
  };

  bool _IsPrimvarEligibleForVertexData(const TfToken& name, const TfToken& role)
  {
    if (name == HdTokens->normals ||
        name == _tokens->tangents ||
        name == _tokens->bitangentSigns ||
        role == HdPrimvarRoleTokens->textureCoordinate)
    {
      return true;
    }

    for (const TfToken& t : _texcoordPrimvarNameHints)
    {
      if (name == t)
      {
        return true;
      }
    }

    return false;
  }
}

HdGatlingMesh::HdGatlingMesh(const SdfPath& id,
                             GiScene* scene,
                             const GiMaterial* defaultMaterial)
  : HdMesh(id)
  , _scene(scene)
  , _defaultMaterial(defaultMaterial)
{
}

HdGatlingMesh::~HdGatlingMesh()
{
  if (_baseMesh)
  {
    giDestroyMesh(_baseMesh);
  }
  for (GiMesh* m : _subMeshes)
  {
    giDestroyMesh(m);
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

  bool updateGeometry =
    (*dirtyBits & HdChangeTracker::DirtyPoints) |
    (*dirtyBits & HdChangeTracker::DirtyNormals) |
    (*dirtyBits & HdChangeTracker::DirtyPrimvar) |
    (*dirtyBits & HdChangeTracker::DirtyTopology);

  if (updateGeometry)
  {
    if (_baseMesh)
    {
      giDestroyMesh(_baseMesh);
      _baseMesh = nullptr;
    }
    for (GiMesh* m : _subMeshes)
    {
      giDestroyMesh(m);
    }
    _subMeshes.clear();

    _CreateGiMeshes(sceneDelegate);

    (*dirtyBits) |= HdChangeTracker::DirtyMaterialId; // force material assignment
  }

  if (!_baseMesh && _subMeshes.empty())
  {
    return;
  }

  if (*dirtyBits & HdChangeTracker::DirtyVisibility)
  {
    _UpdateVisibility(sceneDelegate, &dirtyBitsCopy);

    if (_baseMesh)
    {
      giSetMeshVisibility(_baseMesh, sceneDelegate->GetVisible(id));
    }
    for (GiMesh* m : _subMeshes)
    {
      giSetMeshVisibility(m, sceneDelegate->GetVisible(id));
    }
  }

  if ((*dirtyBits & HdChangeTracker::DirtyInstancer) || (*dirtyBits & HdChangeTracker::DirtyInstanceIndex))
  {
    _UpdateInstancer(sceneDelegate, &dirtyBitsCopy);

    const SdfPath& instancerId = GetInstancerId();

    HdInstancer::_SyncInstancerAndParents(renderIndex, instancerId);

    VtMatrix4fArray transforms;
    std::vector<GiPrimvarData> instancerPrimvars;

    if (instancerId.IsEmpty())
    {
      transforms.resize(1);
      transforms[0] = GfMatrix4f(1.0);
    }
    else
    {
      HdInstancer* boxedInstancer = renderIndex.GetInstancer(instancerId);
      HdGatlingInstancer* instancer = static_cast<HdGatlingInstancer*>(boxedInstancer);

      transforms = instancer->ComputeFlattenedTransforms(id);
      instancerPrimvars = instancer->ComputeFlattenedPrimvars(id);
    }

    auto transformsSize = uint32_t(transforms.size());
    auto transformsData = (const float(*)[4][4]) transforms[0].data();

    if (_baseMesh)
    {
      giSetMeshInstanceTransforms(_baseMesh, transformsSize, transformsData);
      giSetMeshInstancerPrimvars(_baseMesh, instancerPrimvars);
    }
    for (GiMesh* m : _subMeshes)
    {
      giSetMeshInstanceTransforms(m, transformsSize, transformsData);
      giSetMeshInstancerPrimvars(m, instancerPrimvars);
    }
  }

  if (*dirtyBits & HdChangeTracker::DirtyTransform)
  {
    auto transform = GfMatrix4f(sceneDelegate->GetTransform(id));

    if (_baseMesh)
    {
      giSetMeshTransform(_baseMesh, transform.data());
    }
    for (GiMesh* m : _subMeshes)
    {
      giSetMeshTransform(m, transform.data());
    }
  }

  if ((*dirtyBits & HdChangeTracker::DirtyMaterialId) && _baseMesh)
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

    giSetMeshMaterial(_baseMesh, giMat);
  }

  *dirtyBits = HdChangeTracker::Clean;
}


void HdGatlingMesh::_AnalyzePrimvars(HdSceneDelegate* sceneDelegate,
                                     bool& foundNormals,
                                     bool& indexingAllowed)
{
  const SdfPath& id = GetId();

  foundNormals = false;
  indexingAllowed = true;

  for (int i = 0; i < int(HdInterpolationCount); i++)
  {
    const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, (HdInterpolation) i);

    for (const HdPrimvarDescriptor& primvar : primvarDescs)
    {
      VtValue value = GetPrimvar(sceneDelegate, primvar.name);

      if (!HdGatlingIsPrimvarTypeSupported(value))
      {
        continue;
      }

      if (primvar.interpolation == HdInterpolationFaceVarying)
      {
        indexingAllowed = false;
      }

      if (primvar.name == HdTokens->normals)
      {
        foundNormals = true;
      }
    }
  }
}

std::optional<HdGatlingMesh::ProcessedPrimvar> HdGatlingMesh::_ProcessPrimvar(HdSceneDelegate* sceneDelegate,
                                                                              const VtIntArray& primitiveParams,
                                                                              const HdPrimvarDescriptor& primvarDesc,
                                                                              const VtVec3iArray& faces,
                                                                              uint32_t vertexCount,
                                                                              bool indexingAllowed,
                                                                              bool forceVertexInterpolation)
{
  const SdfPath& id = GetId();

  VtValue boxedValues = GetPrimvar(sceneDelegate, primvarDesc.name);
  HdType type = HdGetValueTupleType(boxedValues).type;

  if (!HdGatlingIsPrimvarTypeSupported(boxedValues))
  {
    return std::nullopt;
  }

  // Gi doesn't natively support bool primvars; convert them to ints
  if (type == HdTypeBool)
  {
    HdGatlingConvertVtBoolArrayToVtIntArray(boxedValues);
    type = HdTypeInt32;
  }

  VtValue result = boxedValues;
  HdVtBufferSource buffer(primvarDesc.name, boxedValues);

  if ((primvarDesc.interpolation == HdInterpolationVertex ||
      // Varying is equivalent to Vertex for non-subdivided polygonal surfaces (and we don't support subdivision):
      // https://github.com/usd-wg/assets/tree/907d5f17bbe933fc14441a3f3ab69a5bd8abe32a/docs/PrimvarInterpolation#vertex
      primvarDesc.interpolation == HdInterpolationVarying) && !indexingAllowed)
  {
    result = _DeindexBufferElements(type, buffer, faces);
  }
  else if (primvarDesc.interpolation == HdInterpolationConstant && forceVertexInterpolation)
  {
    result = _ExpandBufferElements(buffer, type, vertexCount);
  }
  else if (primvarDesc.interpolation == HdInterpolationFaceVarying)
  {
    TF_AXIOM(!indexingAllowed);

    HdMeshTopology topology = GetMeshTopology(sceneDelegate);
    HdMeshUtil meshUtil(&topology, id);
    if (!meshUtil.ComputeTriangulatedFaceVaryingPrimvar(buffer.GetData(),
                                                        buffer.GetNumElements(),
                                                        type,
                                                        &result))
    {
      return std::nullopt;
    }
  }
  else if (primvarDesc.interpolation == HdInterpolationUniform)
  {
    result = _CreateSizedArray(type, faces.size() * (forceVertexInterpolation ? 3 : 1));

    uint8_t* srcPtr = (uint8_t*) HdGetValueData(boxedValues);
    uint8_t* dstPtr = (uint8_t*) HdGetValueData(result);
    size_t elementSize = HdDataSizeOfType(type);

    for (size_t faceIndex = 0; faceIndex < faces.size(); faceIndex++)
    {
      int oldFaceIndex = HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(primitiveParams[faceIndex]);

      TF_DEV_AXIOM(oldFaceIndex < boxedValues.GetArraySize());

      if (forceVertexInterpolation)
      {
        memcpy(&dstPtr[(faceIndex * 3 + 0) * elementSize], &srcPtr[oldFaceIndex * elementSize], elementSize);
        memcpy(&dstPtr[(faceIndex * 3 + 1) * elementSize], &srcPtr[oldFaceIndex * elementSize], elementSize);
        memcpy(&dstPtr[(faceIndex * 3 + 2) * elementSize], &srcPtr[oldFaceIndex * elementSize], elementSize);
      }
      else
      {
        memcpy(&dstPtr[faceIndex * elementSize], &srcPtr[oldFaceIndex * elementSize], elementSize);
      }
    }
  }
  else if (primvarDesc.interpolation == HdInterpolationInstance && forceVertexInterpolation)
  {
    TF_WARN("Interpolation mode 'instance' unsupported for primary primvar %s", id.GetText());
    return std::nullopt;
  }

  return ProcessedPrimvar{
    .interpolation = forceVertexInterpolation ? HdInterpolationVertex : primvarDesc.interpolation,
    .type = type,
    .role = primvarDesc.role,
    .indexMatchingData = result
  };
}

HdGatlingMesh::PrimvarMap HdGatlingMesh::_ProcessPrimvars(HdSceneDelegate* sceneDelegate,
                                                          const VtIntArray& primitiveParams,
                                                          const VtVec3iArray& faces,
                                                          uint32_t vertexCount,
                                                          bool indexingAllowed)
{
  PrimvarMap map;

  for (int i = 0; i < int(HdInterpolationCount); i++)
  {
    const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, (HdInterpolation) i);

    for (const HdPrimvarDescriptor& primvar : primvarDescs)
    {
      const TfToken& name = primvar.name;

      if (name.IsEmpty() || name == HdTokens->points)
      {
        continue;
      }

      // Force primvars that could be baked into vertices to be vertex-interpolated
      bool forceVertexInterpolation = _IsPrimvarEligibleForVertexData(name, primvar.role);

      auto p = _ProcessPrimvar(sceneDelegate, primitiveParams, primvar, faces, vertexCount, indexingAllowed, forceVertexInterpolation);
      if (p.has_value())
      {
        map[name] = *p;
      }
    }
  }

  return map;
}

std::vector<GiPrimvarData> HdGatlingMesh::_CollectSecondaryPrimvars(const PrimvarMap& primvarMap)
{
  const SdfPath& id = GetId();

  std::vector<GiPrimvarData> result;

  for (auto it = primvarMap.begin(); it != primvarMap.end(); it++)
  {
    const TfToken& name = it->first;
    const ProcessedPrimvar& p = it->second;

    if (name.IsEmpty() ||
#if 0
        // FIXME: optimization disabled, see MtlxDocumentPatcher.cpp
        name == _tokens->st || name == _tokens->st0 || name == _tokens->st_0 ||
        name == _tokens->UV0 || name == _tokens->tangents ||
#endif
        name == HdTokens->points || name == HdTokens->normals)
    {
      continue;
    }

    uint32_t dataSize = HdDataSizeOfTupleType(HdGetValueTupleType(p.indexMatchingData));

    std::vector<uint8_t> data(dataSize);
    const uint8_t* srcPtr = (uint8_t*) HdGetValueData(p.indexMatchingData);
    memcpy(&data[0], srcPtr, data.size());

    GiPrimvarInterpolation interpolation;
    if (p.interpolation == HdInterpolationConstant) {
      interpolation = GiPrimvarInterpolation::Constant;
    }
    else if (p.interpolation == HdInterpolationInstance) {
      interpolation = GiPrimvarInterpolation::Instance;
    }
    else if (p.interpolation == HdInterpolationUniform) {
      interpolation = GiPrimvarInterpolation::Uniform;
    }
    else {
      interpolation = GiPrimvarInterpolation::Vertex;
    }

    result.push_back(GiPrimvarData{
      .name = name.GetString(),
      .type = HdGatlingGetGiPrimvarType(p.type),
      .interpolation = interpolation,
      .data = data
    });
  }

  return result;
}

void HdGatlingMesh::_CreateGiMeshes(HdSceneDelegate* sceneDelegate)
{
  const SdfPath& id = GetId();

  VtVec3iArray faces;

  // Faces
  const HdMeshTopology& topology = GetMeshTopology(sceneDelegate);
  HdMeshUtil meshUtil(&topology, id);

  VtIntArray primitiveParams;
  meshUtil.ComputeTriangleIndices(&faces, &primitiveParams);
  auto faceCount = uint32_t(faces.size());

  // Points (required; one per vertex)
  VtValue boxedPoints = GetPoints(sceneDelegate);

  if (boxedPoints.IsEmpty() || !boxedPoints.IsHolding<VtVec3fArray>())
  {
    TF_RUNTIME_ERROR("Points primvar not found (%s)", id.GetText());
    return;
  }

  // Analyze primvars
  bool foundNormals;
  bool useIndexing;
  _AnalyzePrimvars(sceneDelegate, foundNormals, useIndexing);

  // Generate fallback normals on original points
  VtVec3fArray normals;
  if (!foundNormals)
  {
    VtVec3fArray points = boxedPoints.UncheckedGet<VtVec3fArray>();

    Hd_VertexAdjacency adjacency;
    adjacency.BuildAdjacencyTable(&topology);
    normals = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, points.size(), points.cdata());
    TF_AXIOM(normals.size() == points.size());

    if (!useIndexing)
    {
      HdVtBufferSource buffer(HdTokens->normals, VtValue(normals));
      normals = _DeindexBufferElements(HdTypeFloatVec3, buffer, faces).Get<VtVec3fArray>();
    }
  }

  // Deindex points and process primvars
  if (!useIndexing)
  {
    HdVtBufferSource buffer(HdTokens->points, boxedPoints);
    boxedPoints = _DeindexBufferElements(HdTypeFloatVec3, buffer, faces);
  }
  VtVec3fArray points = boxedPoints.UncheckedGet<VtVec3fArray>();

  PrimvarMap primvarMap = _ProcessPrimvars(sceneDelegate, primitiveParams, faces, points.size(), useIndexing);

  // Use normals if authored
  if (foundNormals)
  {
    auto normalsIt = primvarMap.find(HdTokens->normals);
    TF_AXIOM(normalsIt != primvarMap.end());

    const ProcessedPrimvar& pn = normalsIt->second;
    TF_VERIFY(pn.type == HdTypeFloatVec3);

    normals = pn.indexMatchingData.Get<VtVec3fArray>();
  }

  // Texcoords. Find primary primvar by name and role.
  TfToken texcoordPrimvarName;
  for (const TfToken& name : _texcoordPrimvarNameHints)
  {
    if (primvarMap.count(name) > 0)
    {
      texcoordPrimvarName = name;
      break;
    }
  }

  if (texcoordPrimvarName.IsEmpty())
  {
    for (auto it = primvarMap.begin(); it != primvarMap.end(); it++)
    {
      if (it->second.role == HdPrimvarRoleTokens->textureCoordinate)
      {
        texcoordPrimvarName = it->first;
      }
    }
  }

  VtVec2fArray texCoords;
  if (!texcoordPrimvarName.IsEmpty())
  {
    const ProcessedPrimvar& pt = primvarMap[texcoordPrimvarName];
    TF_VERIFY(pt.type == HdTypeFloatVec2);

    texCoords = pt.indexMatchingData.Get<VtVec2fArray>();
  }

  // Tangents. Although barely used in practice.
  VtVec3fArray tangents;
  VtFloatArray bitangentSigns;

  auto tangentsIt = primvarMap.find(_tokens->tangents);
  if (tangentsIt != primvarMap.end())
  {
    const ProcessedPrimvar& pt = tangentsIt->second;

    if (pt.type == HdTypeFloatVec4)
    {
      VtVec4fArray vec4Tangents = pt.indexMatchingData.Get<VtVec4fArray>();
      tangents.resize(vec4Tangents.size());
      bitangentSigns.resize(vec4Tangents.size());

      for (size_t i = 0; i < vec4Tangents.size(); i++)
      {
        tangents[i] = GfVec3f(vec4Tangents[i].data());
        bitangentSigns[i] = vec4Tangents[i][3];
      }
    }
    else if (pt.type == HdTypeFloatVec3)
    {
      tangents = pt.indexMatchingData.Get<VtVec3fArray>();

      auto bitangentSignsIt = primvarMap.find(_tokens->bitangentSigns);
      if (bitangentSignsIt != primvarMap.end())
      {
        const ProcessedPrimvar& pb = tangentsIt->second;

        if (pb.type == HdTypeFloat)
        {
          bitangentSigns = pb.indexMatchingData.Get<VtFloatArray>();
        }
      }
    }
    else
    {
      TF_WARN("Invalid tangents type for %s", id.GetText());
    }
  }

  // Deindex faces
  if (!useIndexing)
  {
    for (uint32_t i = 0; i < faceCount; i++)
    {
      faces[i] = GfVec3i(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }
  }

  // Collect vertices and indices
  _VertexStreams s = {
    .faces = faces,
    .points = points,
    .normals = normals,
    .texCoords = texCoords,
    .tangents = tangents,
    .bitangentSigns = bitangentSigns,
  };

  std::vector<GiFace> giFaces;
  std::vector<GiVertex> giVertices;
  _BakeMeshGeometry(s, giFaces, giVertices);

  // Collect secondary primvars
  std::vector<GiPrimvarData> secondaryPrimvars = _CollectSecondaryPrimvars(primvarMap);

  // Collect geometry subset data
  size_t oldFaceCount = topology.GetNumFaces();
  std::vector<int> oldFaceOwnership(oldFaceCount, 0/*base mesh index*/);

  const HdGeomSubsets& geomSubsets = topology.GetGeomSubsets();
  for (size_t i = 0; i < geomSubsets.size(); i++)
  {
    const HdGeomSubset& subset = geomSubsets[i];

    for (int d : subset.indices)
    {
      if (d >= oldFaceOwnership.size())
      {
        std::string subsetName = subset.id.GetName();
        TF_WARN("GeomSubset %s has invalid face index %i", subsetName.c_str(), d);
        continue;
      }
      oldFaceOwnership[d] = i + 1; // 0 is base mesh
    }
  }

  struct SubMeshData
  {
    std::vector<GiFace> faces;
    std::vector<int> faceIds;
    int maxFaceId = -1;
  };
  std::vector<SubMeshData> subMeshData(geomSubsets.size() + 1/*base mesh*/);

  for (auto& d : subMeshData)
  {
    d.faces.reserve(faceCount);
    d.faceIds.reserve(faceCount);
  }

  for (size_t i = 0; i < faceCount; i++)
  {
    int oldFaceId = HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(primitiveParams[i]);
    int owner = oldFaceOwnership[oldFaceId];

    SubMeshData& subMesh = subMeshData[owner];
    subMesh.faces.push_back(giFaces[i]);
    subMesh.faceIds.push_back(oldFaceId);
    subMesh.maxFaceId = std::max(subMesh.maxFaceId, oldFaceId);
  }

  // Create submeshes
  TfToken orientation = topology.GetOrientation();
  bool isLeftHanded = (orientation == _tokens->leftHanded);

  auto createMesh = [&](const SubMeshData& subMesh) -> GiMesh* {
    if (subMesh.maxFaceId == -1)
    {
      return nullptr;
    }

    GiMeshDesc desc = {
      .faces = subMesh.faces,
      .faceIds = subMesh.faceIds,
      .id = GetPrimId(),
      .isLeftHanded = isLeftHanded,
      .name = id.GetText(),
      .maxFaceId = (uint32_t) subMesh.maxFaceId,
      .primvars = secondaryPrimvars,
      .vertices = giVertices,
    };

    return giCreateMesh(_scene, desc);
  };

  HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();

  for (size_t i = 0; i < subMeshData.size(); i++)
  {
    const auto& data = subMeshData[i];

    GiMesh* subMesh = createMesh(data);

    if (i == 0)
    {
      _baseMesh = subMesh;
    }
    else if (subMesh)
    {
      _subMeshes.push_back(subMesh);

      // Assign material to GeomSubset. Unlike the mesh material, changes invalidate the topology, causing a full rebuild.
      const HdGeomSubset& geomSubset = geomSubsets[i - 1];

      auto* materialPrim = static_cast<HdGatlingMaterial*>(renderIndex.GetSprim(HdPrimTypeTokens->material, geomSubset.materialId));

      const GiMaterial* giMat = materialPrim ? materialPrim->GetGiMaterial() : nullptr;

      if (!giMat)
      {
        giMat = _defaultMaterial;
      }

      giSetMeshMaterial(subMesh, giMat);
    }
  }
}

HdDirtyBits HdGatlingMesh::GetInitialDirtyBitsMask() const
{
  return HdChangeTracker::DirtyPoints |
         HdChangeTracker::DirtyNormals |
         HdChangeTracker::DirtyPrimvar |
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
