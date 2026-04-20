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

#include "particleField.h"
#include "utils.h"

#include <gtl/gb/Fmt.h>
#include <gtl/gi/Gi.h>

#include <pxr/base/gf/matrix4f.h>
#include <pxr/usd/usdVol/tokens.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
  // Exported from Blender, icosphere with subdivision level 1, smooth vertex normals
  const std::vector<GiFace> ICOSAHEDRON_FACES = {
{{ 0, 1, 2 }}, {{ 1, 0, 5 }}, {{ 0, 2, 3 }}, {{ 0, 3, 4 }}, {{ 0, 4, 5 }}, {{ 1, 5, 10 }}, {{ 2, 1, 6 }}, {{ 3, 2, 7 }}, {{ 4, 3, 8 }}, {{ 5, 4, 9 }}, {{ 1, 10, 6 }}, {{ 2, 6, 7 }}, {{ 3, 7, 8 }}, {{ 4, 8, 9 }}, {{ 5, 9, 10 }}, {{ 6, 10, 11 }}, {{ 7, 6, 11 }}, {{ 8, 7, 11 }}, {{ 9, 8, 11 }}, {{ 10, 9, 11 }}
  };
  const std::vector<GiVertex> ICOSAHEDRON_VERTICES = {
{{ 0.000000f, -1.000000f, 0.000000f }, 0.0f, { -0.0000f, -1.0000f, -0.0000f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ 0.723600f, -0.447215f, 0.525720f }, 0.0f, { 0.7236f, -0.4472f, 0.5257f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ -0.276385f, -0.447215f, 0.850640f }, 0.0f, { -0.2764f, -0.4472f, 0.8507f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ -0.894425f, -0.447215f, 0.000000f }, 0.0f, { -0.8944f, -0.4472f, -0.0000f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ -0.276385f, -0.447215f, -0.850640f }, 0.0f, { -0.2764f, -0.4472f, -0.8507f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ 0.723600f, -0.447215f, -0.525720f }, 0.0f, { 0.7236f, -0.4472f, -0.5257f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ 0.276385f, 0.447215f, 0.850640f }, 0.0f, { 0.2764f, 0.4472f, 0.8507f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ -0.723600f, 0.447215f, 0.525720f }, 0.0f, { -0.7236f, 0.4472f, 0.5257f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ -0.723600f, 0.447215f, -0.525720f }, 0.0f, { -0.7236f, 0.4472f, -0.5257f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ 0.276385f, 0.447215f, -0.850640f }, 0.0f, { 0.2764f, 0.4472f, -0.8507f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ 0.894425f, 0.447215f, 0.000000f }, 0.0f, { 0.8944f, 0.4472f, -0.0000f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f },
{{ 0.000000f, 1.000000f, 0.000000f }, 0.0f, { -0.0000f, 1.0000f, -0.0000f }, 0.0f, { 0.0f, 1.0f, 0.0f }, 1.0f }
  };
}

HdGatlingParticleField::HdGatlingParticleField(const SdfPath& id, GiScene* scene, GiMaterial* sh0Mat)
  : HdRprim(id)
  , _giScene(scene)
  , _sh0Mat(sh0Mat)
{
}

HdGatlingParticleField::~HdGatlingParticleField()
{
  if (_ellipseMesh)
  {
    giDestroyMesh(_ellipseMesh);
  }
}

HdDirtyBits HdGatlingParticleField::GetInitialDirtyBitsMask() const
{
  return HdChangeTracker::DirtyPoints |
         HdChangeTracker::DirtyWidths |
         HdChangeTracker::DirtyPrimvar |
         HdChangeTracker::DirtyTransform;
}

void HdGatlingParticleField::Sync(HdSceneDelegate* sceneDelegate,
                                  HdRenderParam* renderParam,
                                  HdDirtyBits* dirtyBits,
                                  const TfToken& reprToken)
{
  const SdfPath& id = GetId();

  if (!HdChangeTracker::IsAnyPrimvarDirty(*dirtyBits, id))
  {
    return;
  }

  if (_ellipseMesh)
  {
    giDestroyMesh(_ellipseMesh);
  }

  // Extract SH degree
  int shDegree = -1;
  {
    const auto& descs = sceneDelegate->GetPrimvarDescriptors(id, HdInterpolationConstant);

    for (const HdPrimvarDescriptor& primvar : descs)
    {
      const TfToken& name = primvar.name;

      VtValue values = sceneDelegate->Get(id, primvar.name);

      if (name != UsdVolTokens->radianceSphericalHarmonicsDegree)
      {
        continue;
      }

      if (!values.IsHolding<int>())
      {
        TF_CODING_ERROR("unsupported SH coef degree type");
        return;
      }

      shDegree = values.UncheckedGet<int>();
    }
  }

  if (shDegree == -1)
  {
    TF_CODING_ERROR("non-GS particle field %s not supported", id.GetText());
    return;
  }

  // Extract primvars
  VtValue boxedPositions;
  VtValue boxedOrientations;
  VtValue boxedScales;
  VtValue boxedOpacities;
  VtValue boxedShCoefs;

  const auto& primvarDescs = sceneDelegate->GetPrimvarDescriptors(id, HdInterpolationVertex);

  for (const HdPrimvarDescriptor& primvar : primvarDescs)
  {
    const TfToken& name = primvar.name;

    VtValue values = sceneDelegate->Get(id, primvar.name);

    if (name == UsdVolTokens->positions)
    {
      boxedPositions = values;
      continue;
    }
    else if (name == UsdVolTokens->orientations)
    {
      boxedOrientations = values;
      continue;
    }
    else if (name == UsdVolTokens->scales)
    {
      boxedScales = values;
      continue;
    }
    else if (name == UsdVolTokens->opacities)
    {
      boxedOpacities = values;
      continue;
    }
    // TODO: support radianceSphericalHarmonicsCoefficientsh
    else if (name == UsdVolTokens->radianceSphericalHarmonicsCoefficients)
    {
      boxedShCoefs = values;
      continue;
    }

    TF_WARN("unsupported primvar %s:%s", id.GetText(), name.GetText());
  }

  if (boxedPositions.IsEmpty() || boxedOrientations.IsEmpty() ||
      boxedScales.IsEmpty() || boxedShCoefs.IsEmpty())
  {
    TF_RUNTIME_ERROR("required primvar missing on GS %s", id.GetText());
    return;
  }

  VtVec3dArray positions;
  VtQuatdArray orientations;
  VtVec3dArray scales;
  if (!HdGatlingUnboxPRSPrimvars(boxedPositions,
                                 boxedOrientations,
                                 boxedScales,
                                 positions,
                                 orientations,
                                 scales))
  {
    return;
  }

  if (!boxedShCoefs.IsHolding<VtVec3fArray>() || !boxedOpacities.IsHolding<VtFloatArray>())
  {
    TF_RUNTIME_ERROR("invalid GS primvar types");
    return;
  }

  uint32_t totalCoefCount = (shDegree + 1) * (shDegree + 1);

  // (flat layout: SH > degree > rgb tuple > float)
  VtVec3fArray shCoefs = boxedShCoefs.UncheckedGet<VtVec3fArray>();
  if (shCoefs.size() != (positions.size() * totalCoefCount))
  {
    TF_RUNTIME_ERROR("malformed SH coefs primvar");
    return;
  }

  std::vector<GiPrimvarData> primvars;
  primvars.reserve(totalCoefCount);

  for (uint32_t sh = 0, coefOffset = 0; sh < uint32_t(shDegree + 1); sh++)
  {
    uint32_t coefCount = (sh == 0) ? 1 : (sh * 2 + 1);

    for (uint32_t c = 0; c < coefCount; c++)
    {
      VtVec3fArray coefs(positions.size());

      for (size_t i = 0; i < coefs.size(); i++)
      {
        coefs[i] = shCoefs[i * totalCoefCount + coefOffset + c];
      }

      std::vector<uint8_t> data(coefs.size() * sizeof(GfVec3f));
      memcpy(&data[0], coefs.data(), data.size());

      primvars.push_back(GiPrimvarData{
        .name = GB_FMT("__gtl_gs_sh{}c{}", sh, c),
        .type = GiPrimvarType::Vec3,
        .interpolation = GiPrimvarInterpolation::Instance,
        .data = data
      });
    }

    coefOffset += coefCount;
  }

  VtFloatArray opacities = boxedOpacities.UncheckedGet<VtFloatArray>();

  if (opacities.size() == positions.size())
  {
    std::vector<uint8_t> data(opacities.size() * sizeof(float));
    memcpy(&data[0], opacities.data(), data.size());

    primvars.push_back(GiPrimvarData{
      .name = "__gtl_gs_opacity",
      .type = GiPrimvarType::Float,
      .interpolation = GiPrimvarInterpolation::Instance,
      .data = data
    });
  }
  else
  {
    TF_RUNTIME_ERROR("malformed opacities primvar");
    return;
  }

  // Calculate instance transforms
  VtIntArray indices;
  indices.resize(positions.size());
  for (size_t i = 0; i < indices.size(); i++)
  {
    indices[i] = i;
  }

  GfMatrix4d transform = sceneDelegate->GetTransform(id);
  VtMatrix4dArray instanceTransforms; // optional
  VtMatrix4fArray outTransforms;

// TODO: clean up the code in this method
VtVec3fArray origScales(positions.size());

  // Enlarge splats according to the heuristic outlined in
  // (Eq. 7) Moenne-Loccoz et al. 2024 - 3D Gaussian Ray Tracing
  const double invAlphaMin = 1.0 / 0.01;
  for (size_t i = 0; i < scales.size(); i++)
  {
    origScales[i] = GfVec3f(scales[i]);
    scales[i] *= sqrt(2.0 * log(opacities[i] * invAlphaMin));
  }

  HdGatlingPRSToTransforms(indices,
                           transform,
                           instanceTransforms,
                           positions,
                           orientations,
                           scales,
                           outTransforms);

  {
    VtVec3fArray positionsWS(positions.size());
    for (size_t i = 0; i < positions.size(); i++)
    {
      positionsWS[i] = GfVec3f(positions[i]);
    }

    std::vector<uint8_t> data(positionsWS.size() * sizeof(GfVec3f));
    memcpy(&data[0], positionsWS.data(), data.size());

    primvars.push_back(GiPrimvarData{
      .name = "__gtl_gs_pos",
      .type = GiPrimvarType::Vec3,
      .interpolation = GiPrimvarInterpolation::Instance,
      .data = data
    });
  }
  {
VtVec3fArray scalesF(scales.size());
for (size_t i = 0; i < scales.size(); i++)
{
  scalesF[i] = GfVec3f(scales[i]);
}

    std::vector<uint8_t> data(scalesF.size() * sizeof(GfVec3f));
    memcpy(&data[0], scalesF.data(), data.size());

    primvars.push_back(GiPrimvarData{
      .name = "__gtl_gs_scale",
      .type = GiPrimvarType::Vec3,
      .interpolation = GiPrimvarInterpolation::Instance,
      .data = data
    });
  }
  {
VtVec4fArray rotsF(orientations.size());
VtVec3fArray rotsIF(orientations.size());
VtFloatArray rotsRF(orientations.size());

for (size_t i = 0; i < orientations.size(); i++)
{
  auto q = orientations[i];

  // bake root transform into splat rot (?)
//q = transform.ExtractRotationQuat() * q;

//q = q.GetInverse();

  rotsIF[i] = GfVec3f(
    q.GetImaginary()[0],
    q.GetImaginary()[1],
    q.GetImaginary()[2]
  );
  rotsRF[i] = float(
    q.GetReal()
  );
  rotsF[i] = GfVec4f(
    q.GetImaginary()[0],
    q.GetImaginary()[1],
    q.GetImaginary()[2],
    q.GetReal()
  );
}

{
    std::vector<uint8_t> data(rotsF.size() * sizeof(GfVec4f));
    memcpy(&data[0], rotsF.data(), data.size());

    primvars.push_back(GiPrimvarData{
      .name = "__gtl_gs_rots",
      .type = GiPrimvarType::Vec4,
      .interpolation = GiPrimvarInterpolation::Instance,
      .data = data
    });
}
{
    std::vector<uint8_t> data(rotsIF.size() * sizeof(GfVec3f));
    memcpy(&data[0], rotsIF.data(), data.size());

    primvars.push_back(GiPrimvarData{
      .name = "__gtl_gs_rotsI",
      .type = GiPrimvarType::Vec3,
      .interpolation = GiPrimvarInterpolation::Instance,
      .data = data
    });
}
{
    std::vector<uint8_t> data(rotsRF.size() * sizeof(float));
    memcpy(&data[0], rotsRF.data(), data.size());

    primvars.push_back(GiPrimvarData{
      .name = "__gtl_gs_rotsR",
      .type = GiPrimvarType::Float,
      .interpolation = GiPrimvarInterpolation::Instance,
      .data = data
    });
}
}

  // Create mesh with instances
  std::vector<int> faceIds;
  faceIds.resize(ICOSAHEDRON_FACES.size());
  for (int i = 0; i < faceIds.size(); i++)
  {
    faceIds[i] = i;
  }

// TODO: we might need to have the 'no self-hit again' flag
  GiMeshDesc meshDesc = {
    .faceCount = uint32_t(ICOSAHEDRON_FACES.size()),
    .faces = ICOSAHEDRON_FACES,
    .faceIds = faceIds,
    .id = INT32_MAX - 1,
    .isDoubleSided = false,
    .isLeftHanded = false,
    .name = "__gtl_gsellipse",
    .maxFaceId = uint32_t(ICOSAHEDRON_FACES.size() - 1),
    .primvars = primvars,
    .vertexCount = uint32_t(ICOSAHEDRON_VERTICES.size()),
    .vertices = ICOSAHEDRON_VERTICES
  };

  _ellipseMesh = giCreateMesh(_giScene, meshDesc);
  TF_AXIOM(_ellipseMesh);

  auto transformsData = (const float(*)[4][4]) outTransforms[0].data();
  giSetMeshInstanceTransforms(_ellipseMesh, uint32_t(outTransforms.size()), transformsData);
  giSetMeshInstanceIds(_ellipseMesh, uint32_t(indices.size()), indices.data());
  giSetMeshMaterial(_ellipseMesh, _sh0Mat);

  // Check GS properties
  VtValue projectionModeHint = sceneDelegate->Get(id, UsdVolTokens->projectionModeHint);
  VtValue sortingModeHint = sceneDelegate->Get(id, UsdVolTokens->sortingModeHint);

  if (projectionModeHint.IsHolding<TfToken>() &&
      projectionModeHint.UncheckedGet<TfToken>() != UsdVolTokens->perspective)
  {
    TF_WARN("unsupported GS projection mode for prim %s", id.GetText());
  }
  if (sortingModeHint.IsHolding<TfToken>() &&
      sortingModeHint.UncheckedGet<TfToken>() != UsdVolTokens->cameraDistance)
  {
    TF_WARN("unsupported GS sorting mode for prim %s", id.GetText());
  }

  *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

const TfTokenVector& HdGatlingParticleField::GetBuiltinPrimvarNames() const
{
  static TfTokenVector builtins;
  return builtins;
}

void HdGatlingParticleField::_InitRepr(TfToken const& reprToken, HdDirtyBits* dirtyBits)
{
}

HdDirtyBits HdGatlingParticleField::_PropagateDirtyBits(HdDirtyBits bits) const
{
  return bits;
}

PXR_NAMESPACE_CLOSE_SCOPE
