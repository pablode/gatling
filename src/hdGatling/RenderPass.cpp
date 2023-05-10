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

#include "RenderPass.h"
#include "RenderBuffer.h"
#include "Camera.h"
#include "Mesh.h"
#include "Instancer.h"
#include "Material.h"
#include "Tokens.h"
#include "MaterialNetworkTranslator.h"

#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/base/gf/matrix3d.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

std::string _MakeMaterialXColorMaterialSrc(const GfVec3f& color, const char* name)
{
  // Prefer UsdPreviewSurface over MDL diffuse or unlit because we want to give a good first
  // impression (many people will try Pixar's Kitchen scene first), regardless of whether the user
  // is aware of the use or purpose of the displayColor attribute (as opposed to a preview material).
  static const char* USDPREVIEWSURFACE_MTLX_DOC = R"(
    <?xml version="1.0"?>
    <materialx version="1.38">
      <UsdPreviewSurface name="gatling_SR_%s" type="surfaceshader">
        <input name="diffuseColor" type="color3" value="%f, %f, %f" />
      </UsdPreviewSurface>
      <surfacematerial name="gatling_MAT_%s" type="material">
        <input name="surfaceshader" type="surfaceshader" nodename="gatling_SR_%s" />
      </surfacematerial>
    </materialx>
  )";

  return TfStringPrintf(USDPREVIEWSURFACE_MTLX_DOC, name, color[0], color[1], color[2], name, name);
}

HdGatlingRenderPass::HdGatlingRenderPass(HdRenderIndex* index,
                                         const HdRprimCollection& collection,
                                         const HdRenderSettingsMap& settings,
                                         const MaterialNetworkTranslator& materialNetworkTranslator)
  : HdRenderPass(index, collection)
  , m_settings(settings)
  , m_materialNetworkTranslator(materialNetworkTranslator)
  , m_isConverged(false)
  , m_lastSceneStateVersion(UINT32_MAX)
  , m_lastSprimIndexVersion(UINT32_MAX)
  , m_lastRenderSettingsVersion(UINT32_MAX)
  , m_lastVisChangeCount(UINT32_MAX)
  , m_lastBackgroundColor(GfVec4f(0.0f, 0.0f, 0.0f, 0.0f))
  , m_geomCache(nullptr)
  , m_shaderCache(nullptr)
{
  std::string defaultMatSrc = _MakeMaterialXColorMaterialSrc(GfVec3f(1.0f, 0.0f, 1.0f), "invalid");

  m_defaultMaterial = giCreateMaterialFromMtlxStr(defaultMatSrc.c_str());
  TF_AXIOM(m_defaultMaterial);
}

void HdGatlingRenderPass::_ClearMaterials()
{
  for (GiMaterial* mat : m_materials)
  {
    giDestroyMaterial(mat);
  }
  m_materials.clear();
}

HdGatlingRenderPass::~HdGatlingRenderPass()
{
  if (m_geomCache)
  {
    giDestroyGeomCache(m_geomCache);
  }
  if (m_shaderCache)
  {
    giDestroyShaderCache(m_shaderCache);
  }

  giDestroyMaterial(m_defaultMaterial);
  _ClearMaterials();
}

bool HdGatlingRenderPass::IsConverged() const
{
  return m_isConverged;
}

GiVertex _MakeGiVertex(GfMatrix4d transform, GfMatrix4d normalMatrix, const GfVec3f& point, const GfVec3f& normal, const GfVec2f& texCoords)
{
  GfVec3f newPoint = transform.Transform(point);

  GfVec3f newNormal = normalMatrix.TransformDir(normal);
  newNormal.Normalize();

  GiVertex vertex;
  vertex.pos[0] = newPoint[0];
  vertex.pos[1] = newPoint[1];
  vertex.pos[2] = newPoint[2];
  vertex.norm[0] = newNormal[0];
  vertex.norm[1] = newNormal[1];
  vertex.norm[2] = newNormal[2];
  vertex.u = texCoords[0];
  vertex.v = 1.0f - texCoords[1];

  return vertex;
}

void HdGatlingRenderPass::_BakeMeshGeometry(const HdGatlingMesh* mesh,
                                            GfMatrix4d transform,
                                            uint32_t materialIndex,
                                            std::vector<GiFace>& faces,
                                            std::vector<GiVertex>& vertices) const
{
  GfMatrix4d normalMatrix = transform.GetInverse().GetTranspose();

  const VtVec3iArray& meshFaces = mesh->GetFaces();
  const VtVec3fArray& meshPoints = mesh->GetPoints();
  const auto& meshNormals = mesh->GetNormals();
  const auto& meshTexCoords = mesh->GetTexCoords();
  bool isAnyPrimvarNotIndexed = !meshNormals.indexed || !meshTexCoords.indexed;

  uint32_t vertexOffset = vertices.size();

  for (size_t i = 0; i < meshFaces.size(); i++)
  {
    const GfVec3i& vertexIndices = meshFaces[i];

    GiFace face;
    face.v_i[0] = vertexOffset + (isAnyPrimvarNotIndexed ? (i * 3 + 0) : vertexIndices[0]);
    face.v_i[1] = vertexOffset + (isAnyPrimvarNotIndexed ? (i * 3 + 1) : vertexIndices[1]);
    face.v_i[2] = vertexOffset + (isAnyPrimvarNotIndexed ? (i * 3 + 2) : vertexIndices[2]);

    // We always need three unique vertices per face.
    for (size_t j = 0; isAnyPrimvarNotIndexed && j < 3; j++)
    {
      const GfVec3f& point = meshPoints[vertexIndices[j]];
      const GfVec3f& normal = meshNormals.array[meshNormals.indexed ? vertexIndices[j] : (i * 3 + j)];

      bool hasTexCoords = meshTexCoords.array.size() > 0;
      GfVec2f texCoords = hasTexCoords ? meshTexCoords.array[meshTexCoords.indexed ? vertexIndices[j] : (i * 3 + j)] : GfVec2f();

      GiVertex vertex = _MakeGiVertex(transform, normalMatrix, point, normal, texCoords);
      vertices.push_back(vertex);
    }

    faces.push_back(face);
  }

  // Early-out if the vertices are not indexed.
  if (isAnyPrimvarNotIndexed)
  {
    return;
  }

  for (size_t j = 0; j < meshPoints.size(); j++)
  {
    const GfVec3f& point = meshPoints[j];
    const GfVec3f& normal = meshNormals.array[j];

    bool hasTexCoords = meshTexCoords.array.size() > 0;
    GfVec2f texCoords = hasTexCoords ? meshTexCoords.array[j] : GfVec2f();

    GiVertex vertex = _MakeGiVertex(transform, normalMatrix, point, normal, texCoords);
    vertices.push_back(vertex);
  }
}

void HdGatlingRenderPass::_BakeMeshes(HdRenderIndex* renderIndex,
                                      GfMatrix4d rootTransform,
                                      std::vector<const GiMaterial*>& materials,
                                      std::vector<const GiMesh*>& meshes,
                                      std::vector<GiMeshInstance>& instances)
{
  _ClearMaterials();

  TfHashMap<std::string, uint32_t> materialMap;
  materialMap[""] = 0;

  materials.push_back(m_defaultMaterial);

  for (const auto& rprimId : renderIndex->GetRprimIds())
  {
    const HdRprim* rprim = renderIndex->GetRprim(rprimId);

    const HdGatlingMesh* mesh = static_cast<const HdGatlingMesh*>(rprim);
    if (!mesh)
    {
      continue;
    }

    if (!mesh->IsVisible())
    {
      continue;
    }

    VtMatrix4dArray transforms;
    const SdfPath& instancerId = mesh->GetInstancerId();

    if (instancerId.IsEmpty())
    {
      transforms.resize(1);
      transforms[0] = GfMatrix4d(1.0);
    }
    else
    {
      HdInstancer* boxedInstancer = renderIndex->GetInstancer(instancerId);
      HdGatlingInstancer* instancer = static_cast<HdGatlingInstancer*>(boxedInstancer);

      const SdfPath& meshId = mesh->GetId();
      transforms = instancer->ComputeInstanceTransforms(meshId);
    }

    const SdfPath& materialId = mesh->GetMaterialId();
    std::string materialIdStr = materialId.GetAsString();

    uint32_t materialIndex = 0;
    if (!materialId.IsEmpty() && materialMap.find(materialIdStr) != materialMap.end())
    {
      materialIndex = materialMap[materialIdStr];
    }
    else
    {
      HdSprim* sprim = renderIndex->GetSprim(HdPrimTypeTokens->material, materialId);
      HdGatlingMaterial* material = static_cast<HdGatlingMaterial*>(sprim);

      GiMaterial* giMat = nullptr;
      if (material)
      {
        const HdMaterialNetwork2* network = material->GetNetwork();

        if (network)
        {
          giMat = m_materialNetworkTranslator.ParseNetwork(sprim->GetId(), *network);
          m_materials.push_back(giMat);
        }
      }

      if (!giMat && mesh->HasColor())
      {
        // Try to reuse color material by including the RGB value in the name
        const GfVec3f& color = mesh->GetColor();
        materialIdStr = TfStringPrintf("color_%f_%f_%f", color[0], color[1], color[2]);
        std::replace(materialIdStr.begin(), materialIdStr.end(), '.', '_'); // _1.9_ -> _1_9_

        if (materialMap.find(materialIdStr) != materialMap.end())
        {
          materialIndex = materialMap[materialIdStr];
        }
        else
        {
          std::string colorMatSrc = _MakeMaterialXColorMaterialSrc(color, materialIdStr.c_str());
          GiMaterial* giColorMat = giCreateMaterialFromMtlxStr(colorMatSrc.c_str());
          if (giColorMat)
          {
            m_materials.push_back(giColorMat);
            giMat = giColorMat;
          }
        }
      }

      if (giMat)
      {
        materialIndex = materials.size();
        materials.push_back(giMat);
        materialMap[materialIdStr] = materialIndex;
      }
    }

    std::vector<GiFace> faces;
    std::vector<GiVertex> vertices;
    _BakeMeshGeometry(mesh, GfMatrix4d(1.0), materialIndex, faces, vertices);

    GiMeshDesc desc = {0};
    desc.faceCount = faces.size();
    desc.faces = faces.data();
    desc.material = materials[materialIndex];
    desc.vertexCount = vertices.size();
    desc.vertices = vertices.data();

    GiMesh* giMesh = giCreateMesh(&desc);
    assert(giMesh);
    meshes.push_back(giMesh);

    const GfMatrix4d& prototypeTransform = mesh->GetPrototypeTransform();
    for (size_t i = 0; i < transforms.size(); i++)
    {
      GfMatrix4d T = prototypeTransform * transforms[i];

      float instanceTransform[3][4] = {
        (float) T[0][0], (float) T[1][0], (float) T[2][0], (float) T[3][0],
        (float) T[0][1], (float) T[1][1], (float) T[2][1], (float) T[3][1],
        (float) T[0][2], (float) T[1][2], (float) T[2][2], (float) T[3][2]
      };

      GiMeshInstance instance;
      instance.mesh = giMesh;
      memcpy(instance.transform, instanceTransform, sizeof(instanceTransform));
      instances.push_back(instance);
    }
  }
}

void HdGatlingRenderPass::_ConstructGiCamera(const HdGatlingCamera& camera, GiCameraDesc& giCamera) const
{
  // We transform the scene into camera space at the beginning, so for
  // subsequent camera transforms, we need to 'substract' the initial transform.
  GfMatrix4d absInvViewMatrix = camera.GetTransform();
  GfMatrix4d relViewMatrix = absInvViewMatrix * m_rootMatrix;

  GfVec3d position = relViewMatrix.Transform(GfVec3d(0.0, 0.0, 0.0));
  GfVec3d forward = relViewMatrix.TransformDir(GfVec3d(0.0, 0.0, -1.0));
  GfVec3d up = relViewMatrix.TransformDir(GfVec3d(0.0, 1.0, 0.0));

  forward.Normalize();
  up.Normalize();

  giCamera.position[0] = (float) position[0];
  giCamera.position[1] = (float) position[1];
  giCamera.position[2] = (float) position[2];
  giCamera.forward[0] = (float) forward[0];
  giCamera.forward[1] = (float) forward[1];
  giCamera.forward[2] = (float) forward[2];
  giCamera.up[0] = (float) up[0];
  giCamera.up[1] = (float) up[1];
  giCamera.up[2] = (float) up[2];
  giCamera.vfov = camera.GetVFov();
}

const std::unordered_map<TfToken, GiAovId, TfToken::HashFunctor> s_aovIdMappings {
  { HdAovTokens->color,                     GI_AOV_ID_COLOR              },
  { HdAovTokens->normal,                    GI_AOV_ID_NORMAL             },
#ifndef NDEBUG
  { HdGatlingAovTokens->debug_nee,          GI_AOV_ID_DEBUG_NEE          },
  { HdGatlingAovTokens->debug_barycentrics, GI_AOV_ID_DEBUG_BARYCENTRICS },
  { HdGatlingAovTokens->debug_texcoords,    GI_AOV_ID_DEBUG_TEXCOORDS    },
  { HdGatlingAovTokens->debug_bounces,      GI_AOV_ID_DEBUG_BOUNCES      },
  { HdGatlingAovTokens->debug_clock_cycles, GI_AOV_ID_DEBUG_CLOCK_CYCLES },
  { HdGatlingAovTokens->debug_opacity,      GI_AOV_ID_DEBUG_OPACITY      },
#endif
};

const HdRenderPassAovBinding* _FilterAovBinding(const HdRenderPassAovBindingVector& aovBindings)
{
  for (const HdRenderPassAovBinding& aovBinding : aovBindings)
  {
    bool aovSupported = s_aovIdMappings.count(aovBinding.aovName) > 0;

    if (aovSupported)
    {
      return &aovBinding;
    }

    HdGatlingRenderBuffer* renderBuffer = static_cast<HdGatlingRenderBuffer*>(aovBinding.renderBuffer);
    renderBuffer->SetConverged(true);
    continue;
  }

  return nullptr;
}

GiAovId _GetAovId(const TfToken& aovName)
{
  GiAovId id = GI_AOV_ID_COLOR;

  auto iter = s_aovIdMappings.find(aovName);

  if (iter != s_aovIdMappings.end())
  {
    id = iter->second;
  }
  else
  {
    TF_CODING_ERROR(TfStringPrintf("Invalid AOV id %s", aovName.GetText()));
  }

  return id;
}

void HdGatlingRenderPass::_Execute(const HdRenderPassStateSharedPtr& renderPassState,
                                   const TfTokenVector& renderTags)
{
  TF_UNUSED(renderTags);

  m_isConverged = false;

  const auto* camera = static_cast<const HdGatlingCamera*>(renderPassState->GetCamera());
  if (!camera)
  {
    return;
  }

  const HdRenderPassAovBindingVector& aovBindings = renderPassState->GetAovBindings();
  if (aovBindings.empty())
  {
    return;
  }

  const HdRenderPassAovBinding* aovBinding = _FilterAovBinding(aovBindings);
  if (!aovBinding)
  {
    TF_RUNTIME_ERROR("AOV not supported");
    return;
  }

  HdGatlingRenderBuffer* renderBuffer = static_cast<HdGatlingRenderBuffer*>(aovBinding->renderBuffer);
  if (renderBuffer->GetFormat() != HdFormatFloat32Vec4)
  {
    TF_RUNTIME_ERROR("Unsupported render buffer format");
    return;
  }

  HdRenderIndex* renderIndex = GetRenderIndex();
  HdChangeTracker& changeTracker = renderIndex->GetChangeTracker();
  HdRenderDelegate* renderDelegate = renderIndex->GetRenderDelegate();

  GfVec4f backgroundColor(0.0f, 0.0f, 0.0f, 0.0f);
  if (aovBinding->clearValue.IsHolding<GfVec4f>())
  {
    backgroundColor = aovBinding->clearValue.UncheckedGet<GfVec4f>();
  }

  uint32_t sceneStateVersion = changeTracker.GetSceneStateVersion();
  uint32_t sprimIndexVersion = changeTracker.GetSprimIndexVersion();
  uint32_t visibilityChangeCount = changeTracker.GetVisibilityChangeCount();
  uint32_t renderSettingsStateVersion = renderDelegate->GetRenderSettingsVersion();
  GiAovId aovId = _GetAovId(aovBinding->aovName);

  bool sceneChanged = (sceneStateVersion != m_lastSceneStateVersion);
  bool sprimsChanged = (sprimIndexVersion != m_lastSprimIndexVersion);
  bool renderSettingsChanged = (renderSettingsStateVersion != m_lastRenderSettingsVersion);
  bool visibilityChanged = (m_lastVisChangeCount != visibilityChangeCount);
  bool backgroundColorChanged = (backgroundColor != m_lastBackgroundColor);
  bool aovChanged = (aovId != m_lastAovId);

  if (sceneChanged || renderSettingsChanged || visibilityChanged || backgroundColorChanged || aovChanged)
  {
    giInvalidateFramebuffer();
  }

  m_lastSceneStateVersion = sceneStateVersion;
  m_lastSprimIndexVersion = sprimIndexVersion;
  m_lastRenderSettingsVersion = renderSettingsStateVersion;
  m_lastVisChangeCount = visibilityChangeCount;
  m_lastBackgroundColor = backgroundColor;
  m_lastAovId = aovId;

  bool rebuildShaderCache = !m_shaderCache || aovChanged || sprimsChanged /*dome light could have been added/removed*/;
#ifndef NDEBUG
  // HACK: activating the NEE debug render setting requires shader recompilation.
  rebuildShaderCache |= renderSettingsChanged;
#endif
  bool rebuildGeomCache = !m_geomCache || visibilityChanged;

  if (rebuildShaderCache || rebuildGeomCache)
  {
    if (m_shaderCache)
    {
      giDestroyShaderCache(m_shaderCache);
    }
    if (m_geomCache)
    {
      giDestroyGeomCache(m_geomCache);
    }

    const SdfPath& cameraId = camera->GetId();
    printf("rebuilding geom cache for camera %s\n", cameraId.GetText());

    // Transform scene into camera space to increase floating point precision.
    // FIXME: reintroduce and don't apply rotation
    // https://pharr.org/matt/blog/2018/03/02/rendering-in-camera-space
    //GfMatrix4d viewMatrix = camera->GetTransform().GetInverse();
    m_rootMatrix = GfMatrix4d(1.0);// viewMatrix;

    // FIXME: destroy these resources
    std::vector<const GiMaterial*> materials;
    std::vector<const GiMesh*> meshes;
    std::vector<GiMeshInstance> instances;
    _BakeMeshes(renderIndex, m_rootMatrix, materials, meshes, instances);

    GiShaderCacheParams shaderParams;
    shaderParams.aovId = aovId;
    shaderParams.materialCount = materials.size();
    shaderParams.materials = materials.data();

    m_shaderCache = giCreateShaderCache(&shaderParams);
    TF_VERIFY(m_shaderCache, "Unable to create shader cache");

    GiGeomCacheParams geomParams;
    geomParams.meshInstanceCount = instances.size();
    geomParams.meshInstances = instances.data();
    geomParams.shaderCache = m_shaderCache;

    m_geomCache = giCreateGeomCache(&geomParams);
    TF_VERIFY(m_geomCache, "Unable to create geom cache");
  }

  if (!m_geomCache || !m_shaderCache)
  {
    return;
  }

  GiCameraDesc giCamera;
  _ConstructGiCamera(*camera, giCamera);

  GiRenderParams renderParams;
  renderParams.camera = &giCamera;
  renderParams.geomCache = m_geomCache;
  renderParams.shaderCache = m_shaderCache;
  renderParams.imageWidth = renderBuffer->GetWidth();
  renderParams.imageHeight = renderBuffer->GetHeight();
  renderParams.maxBounces = m_settings.find(HdGatlingSettingsTokens->max_bounces)->second.Get<int>();
  renderParams.spp = m_settings.find(HdGatlingSettingsTokens->spp)->second.Get<int>();
  renderParams.rrBounceOffset = m_settings.find(HdGatlingSettingsTokens->rr_bounce_offset)->second.Get<int>();
  // Workaround for bug https://github.com/PixarAnimationStudios/USD/issues/913
  VtValue rr_inv_min_term_prob = m_settings.find(HdGatlingSettingsTokens->rr_inv_min_term_prob)->second;
  VtValue max_sample_value = m_settings.find(HdGatlingSettingsTokens->max_sample_value)->second;
  renderParams.rrInvMinTermProb = float(rr_inv_min_term_prob.Cast<double>().Get<double>());
  renderParams.maxSampleValue = float(max_sample_value.Cast<double>().Get<double>());
  for (uint32_t i = 0; i < 4; i++)
  {
    renderParams.bgColor[i] = backgroundColor[i];
  }

  float* img_data = (float*) renderBuffer->Map();

  int32_t result = giRender(&renderParams, img_data);

  TF_VERIFY(result == GI_OK, "Unable to render scene.");

  renderBuffer->Unmap();

  m_isConverged = true;
}

PXR_NAMESPACE_CLOSE_SCOPE
