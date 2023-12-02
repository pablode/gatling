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
#include "RenderParam.h"
#include "Mesh.h"
#include "Instancer.h"
#include "Material.h"
#include "Tokens.h"
#include "MaterialNetworkCompiler.h"

#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/camera.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
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
}

HdGatlingRenderPass::HdGatlingRenderPass(HdRenderIndex* index,
                                         const HdRprimCollection& collection,
                                         const HdRenderSettingsMap& settings,
                                         const MaterialNetworkCompiler& materialNetworkCompiler,
                                         GiScene* scene)
  : HdRenderPass(index, collection)
  , m_scene(scene)
  , m_settings(settings)
  , m_materialNetworkCompiler(materialNetworkCompiler)
  , m_isConverged(false)
  , m_lastSceneStateVersion(UINT32_MAX)
  , m_lastSprimIndexVersion(UINT32_MAX)
  , m_lastRenderSettingsVersion(UINT32_MAX)
  , m_lastVisChangeCount(UINT32_MAX)
  , m_geomCache(nullptr)
  , m_shaderCache(nullptr)
{
  auto defaultDiffuseColor = GfVec3f(0.18f); // UsdPreviewSurface spec
  std::string defaultMatSrc = _MakeMaterialXColorMaterialSrc(defaultDiffuseColor, "invalid");

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

void HdGatlingRenderPass::_BakeMeshes(HdRenderIndex* renderIndex,
                                      GfMatrix4d rootTransform,
                                      std::vector<const GiMaterial*>& materials,
                                      std::vector<const GiMesh*>& meshes,
                                      std::vector<GiMeshInstance>& instances)
{
  _ClearMaterials();

  TfHashMap<std::string, uint32_t, TfHash> materialMap;
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

    const GiMesh* giMesh = mesh->GetGiMesh();
    if (!giMesh)
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
          giMat = m_materialNetworkCompiler.CompileNetwork(sprim->GetId(), *network);

          if (giMat)
          {
            m_materials.push_back(giMat);
          }
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
      instance.material = materials[materialIndex];
      instance.mesh = giMesh;
      memcpy(instance.transform, instanceTransform, sizeof(instanceTransform));
      instances.push_back(instance);
    }
  }
}

void HdGatlingRenderPass::_ConstructGiCamera(const HdCamera& camera, GiCameraDesc& giCamera, bool clippingEnabled) const
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

  // See https://wiki.panotools.org/Field_of_View
  float aperture = camera.GetVerticalAperture() * GfCamera::APERTURE_UNIT;
  float focalLength = camera.GetFocalLength() * GfCamera::FOCAL_LENGTH_UNIT;
  float vfov = 2.0f * std::atan(aperture / (2.0f * focalLength));

  bool focusOn = true;
#if PXR_VERSION >= 2311
  focusOn = camera.GetFocusOn();
#endif

  giCamera.position[0] = (float) position[0];
  giCamera.position[1] = (float) position[1];
  giCamera.position[2] = (float) position[2];
  giCamera.forward[0] = (float) forward[0];
  giCamera.forward[1] = (float) forward[1];
  giCamera.forward[2] = (float) forward[2];
  giCamera.up[0] = (float) up[0];
  giCamera.up[1] = (float) up[1];
  giCamera.up[2] = (float) up[2];
  giCamera.vfov = vfov;
  giCamera.fStop = float(focusOn) * camera.GetFStop();
  giCamera.focusDistance = camera.GetFocusDistance();
  giCamera.focalLength = focalLength;
  giCamera.clipStart = clippingEnabled ? camera.GetClippingRange().GetMin() : 0.0f;
  giCamera.clipEnd = clippingEnabled ? camera.GetClippingRange().GetMax() : FLT_MAX;
  giCamera.exposure = camera.GetExposure();
}

const std::unordered_map<TfToken, GiAovId, TfToken::HashFunctor> s_aovIdMappings {
  { HdAovTokens->color,                     GI_AOV_ID_COLOR              },
  { HdAovTokens->normal,                    GI_AOV_ID_NORMAL             },
#ifndef NDEBUG
  { HdGatlingAovTokens->debugNee,           GI_AOV_ID_DEBUG_NEE          },
  { HdGatlingAovTokens->debugBarycentrics,  GI_AOV_ID_DEBUG_BARYCENTRICS },
  { HdGatlingAovTokens->debugTexcoords,     GI_AOV_ID_DEBUG_TEXCOORDS    },
  { HdGatlingAovTokens->debugBounces,       GI_AOV_ID_DEBUG_BOUNCES      },
  { HdGatlingAovTokens->debugClockCycles,   GI_AOV_ID_DEBUG_CLOCK_CYCLES },
  { HdGatlingAovTokens->debugOpacity,       GI_AOV_ID_DEBUG_OPACITY      },
  { HdGatlingAovTokens->debugTangents,      GI_AOV_ID_DEBUG_TANGENTS     },
  { HdGatlingAovTokens->debugBitangents,    GI_AOV_ID_DEBUG_BITANGENTS   },
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

  const HdCamera* camera = renderPassState->GetCamera();
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
  HdGatlingRenderParam* renderParam = static_cast<HdGatlingRenderParam*>(renderDelegate->GetRenderParam());

  uint32_t sceneStateVersion = changeTracker.GetSceneStateVersion();
  uint32_t sprimIndexVersion = changeTracker.GetSprimIndexVersion();
  uint32_t visibilityChangeCount = changeTracker.GetVisibilityChangeCount();
  uint32_t renderSettingsStateVersion = renderDelegate->GetRenderSettingsVersion();
  GiAovId aovId = _GetAovId(aovBinding->aovName);

  bool sceneChanged = (sceneStateVersion != m_lastSceneStateVersion);
  bool renderSettingsChanged = (renderSettingsStateVersion != m_lastRenderSettingsVersion);
  bool visibilityChanged = (m_lastVisChangeCount != visibilityChangeCount);
  bool aovChanged = (aovId != m_lastAovId);

  if (sceneChanged || renderSettingsChanged || visibilityChanged || aovChanged)
  {
    giInvalidateFramebuffer();
  }

  m_lastSceneStateVersion = sceneStateVersion;
  m_lastSprimIndexVersion = sprimIndexVersion;
  m_lastRenderSettingsVersion = renderSettingsStateVersion;
  m_lastVisChangeCount = visibilityChangeCount;
  m_lastAovId = aovId;

  bool rebuildShaderCache = !m_shaderCache || aovChanged || giShaderCacheNeedsRebuild() || renderSettingsChanged;

  bool rebuildGeomCache = !m_geomCache || visibilityChanged;

  if (rebuildShaderCache || rebuildGeomCache)
  {
    // Transform scene into camera space to increase floating point precision.
    // FIXME: reintroduce and don't apply rotation
    // https://pharr.org/matt/blog/2018/03/02/rendering-in-camera-space
    //GfMatrix4d viewMatrix = camera->GetTransform().GetInverse();
    m_rootMatrix = GfMatrix4d(1.0);// viewMatrix;

    // FIXME: cache results for shader cache rebuild
    std::vector<const GiMaterial*> materials;
    std::vector<const GiMesh*> meshes;
    std::vector<GiMeshInstance> instances;
    _BakeMeshes(renderIndex, m_rootMatrix, materials, meshes, instances);

    if (rebuildShaderCache)
    {
      if (m_shaderCache)
      {
        giDestroyShaderCache(m_shaderCache);
      }

      auto domeLightCameraVisibilityValueIt = m_settings.find(HdRenderSettingsTokens->domeLightCameraVisibility);

      GiShaderCacheParams shaderParams;
      shaderParams.aovId = aovId;
      shaderParams.depthOfField = m_settings.find(HdGatlingSettingsTokens->depthOfField)->second.Get<bool>();
      shaderParams.domeLightCameraVisible = (domeLightCameraVisibilityValueIt == m_settings.end()) || domeLightCameraVisibilityValueIt->second.GetWithDefault<bool>(true);
      shaderParams.filterImportanceSampling = m_settings.find(HdGatlingSettingsTokens->filterImportanceSampling)->second.Get<bool>();
      shaderParams.materialCount = materials.size();
      shaderParams.materials = materials.data();
      shaderParams.nextEventEstimation = m_settings.find(HdGatlingSettingsTokens->nextEventEstimation)->second.Get<bool>();
      shaderParams.progressiveAccumulation = m_settings.find(HdGatlingSettingsTokens->progressiveAccumulation)->second.Get<bool>();
      shaderParams.scene = m_scene;

      m_shaderCache = giCreateShaderCache(&shaderParams);
      TF_VERIFY(m_shaderCache, "Unable to create shader cache");
    }

    if (m_shaderCache && (rebuildGeomCache || giGeomCacheNeedsRebuild()))
    {
      if (m_geomCache)
      {
        giDestroyGeomCache(m_geomCache);
      }

      GiGeomCacheParams geomParams;
      geomParams.meshInstanceCount = instances.size();
      geomParams.meshInstances = instances.data();
      geomParams.shaderCache = m_shaderCache;

      m_geomCache = giCreateGeomCache(&geomParams);
      TF_VERIFY(m_geomCache, "Unable to create geom cache");
    }
  }

  if (!m_geomCache || !m_shaderCache)
  {
    return;
  }

  GfVec4f backgroundColor = aovBinding->clearValue.GetWithDefault<GfVec4f>(GfVec4f(0.f));

  bool clippingEnabled = renderPassState->GetClippingEnabled() &&
                         m_settings.find(HdGatlingSettingsTokens->clippingPlanes)->second.Get<bool>();

  GiCameraDesc giCamera;
  _ConstructGiCamera(*camera, giCamera, clippingEnabled);

  GiRenderParams renderParams;
  renderParams.camera = &giCamera;
  renderParams.geomCache = m_geomCache;
  renderParams.shaderCache = m_shaderCache;
  renderParams.renderBuffer = renderBuffer->GetGiRenderBuffer();
  renderParams.maxBounces = VtValue::Cast<int>(m_settings.find(HdGatlingSettingsTokens->maxBounces)->second).Get<int>();
  renderParams.spp = VtValue::Cast<int>(m_settings.find(HdGatlingSettingsTokens->spp)->second).Get<int>();
  renderParams.rrBounceOffset = VtValue::Cast<int>(m_settings.find(HdGatlingSettingsTokens->rrBounceOffset)->second).Get<int>();
  renderParams.lightIntensityMultiplier = VtValue::Cast<float>(m_settings.find(HdGatlingSettingsTokens->lightIntensityMultiplier)->second).Get<float>();
  renderParams.rrInvMinTermProb = VtValue::Cast<float>(m_settings.find(HdGatlingSettingsTokens->rrInvMinTermProb)->second).Get<float>();
  renderParams.maxSampleValue = VtValue::Cast<float>(m_settings.find(HdGatlingSettingsTokens->maxSampleValue)->second).Get<float>();
  renderParams.domeLight = renderParam->ActiveDomeLight();
  renderParams.scene = m_scene;
  for (uint32_t i = 0; i < 4; i++)
  {
    renderParams.backgroundColor[i] = backgroundColor[i];
  }

  float* imgData = (float*) renderBuffer->Map();

  int32_t result = giRender(&renderParams, imgData);

  TF_VERIFY(result == GI_OK, "Unable to render scene.");

  renderBuffer->Unmap();

  m_isConverged = true;
}

PXR_NAMESPACE_CLOSE_SCOPE
