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

#include "renderPass.h"
#include "renderBuffer.h"
#include "renderParam.h"
#include "mesh.h"
#include "instancer.h"
#include "material.h"
#include "tokens.h"
#include "materialNetworkCompiler.h"

#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/camera.h>

#include <Gi.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
  const static std::unordered_map<TfToken, GiAovId, TfToken::HashFunctor> _aovIdMappings {
    { HdAovTokens->color,                    GiAovId::Color        },
    { HdAovTokens->normal,                   GiAovId::Normal       },
#ifndef NDEBUG
    { HdGatlingAovTokens->debugNee,          GiAovId::NEE          },
    { HdGatlingAovTokens->debugBarycentrics, GiAovId::Barycentrics },
    { HdGatlingAovTokens->debugTexcoords,    GiAovId::Texcoords    },
    { HdGatlingAovTokens->debugBounces,      GiAovId::Bounces      },
    { HdGatlingAovTokens->debugClockCycles,  GiAovId::ClockCycles  },
    { HdGatlingAovTokens->debugOpacity,      GiAovId::Opacity      },
    { HdGatlingAovTokens->debugTangents,     GiAovId::Tangents     },
    { HdGatlingAovTokens->debugBitangents,   GiAovId::Bitangents   },
    { HdGatlingAovTokens->debugDoubleSided,  GiAovId::DoubleSided  },
#endif
    { HdAovTokens->primId,                   GiAovId::ObjectId     },
  };

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

  const HdRenderPassAovBinding* _FilterAovBinding(const HdRenderPassAovBindingVector& aovBindings)
  {
    for (const HdRenderPassAovBinding& aovBinding : aovBindings)
    {
      bool aovSupported = _aovIdMappings.count(aovBinding.aovName) > 0;

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
    GiAovId id = GiAovId::Color;

    auto iter = _aovIdMappings.find(aovName);

    if (iter != _aovIdMappings.end())
    {
      id = iter->second;
    }
    else
    {
      TF_CODING_ERROR(TfStringPrintf("Invalid AOV id %s", aovName.GetText()));
    }

    return id;
  }
}

HdGatlingRenderPass::HdGatlingRenderPass(HdRenderIndex* index,
                                         const HdRprimCollection& collection,
                                         const HdRenderSettingsMap& settings,
                                         const MaterialNetworkCompiler& materialNetworkCompiler,
                                         GiScene* scene)
  : HdRenderPass(index, collection)
  , _scene(scene)
  , _settings(settings)
  , _materialNetworkCompiler(materialNetworkCompiler)
  , _isConverged(false)
  , _lastSceneStateVersion(UINT32_MAX)
  , _lastSprimIndexVersion(UINT32_MAX)
  , _lastRenderSettingsVersion(UINT32_MAX)
  , _lastVisChangeCount(UINT32_MAX)
  , _geomCache(nullptr)
  , _shaderCache(nullptr)
{
  auto defaultDiffuseColor = GfVec3f(0.18f); // UsdPreviewSurface spec
  std::string defaultMatSrc = _MakeMaterialXColorMaterialSrc(defaultDiffuseColor, "invalid");

  _defaultMaterial = giCreateMaterialFromMtlxStr(defaultMatSrc.c_str());
  TF_AXIOM(_defaultMaterial);
}

void HdGatlingRenderPass::_ClearMaterials()
{
  for (GiMaterial* mat : _materials)
  {
    giDestroyMaterial(mat);
  }
  _materials.clear();
}

HdGatlingRenderPass::~HdGatlingRenderPass()
{
  if (_geomCache)
  {
    giDestroyGeomCache(_geomCache);
  }
  if (_shaderCache)
  {
    giDestroyShaderCache(_shaderCache);
  }

  giDestroyMaterial(_defaultMaterial);
  _ClearMaterials();
}

bool HdGatlingRenderPass::IsConverged() const
{
  return _isConverged;
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

  materials.push_back(_defaultMaterial);

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
          giMat = _materialNetworkCompiler.CompileNetwork(sprim->GetId(), *network);

          if (giMat)
          {
            _materials.push_back(giMat);
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
            _materials.push_back(giColorMat);
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
  GfMatrix4d relViewMatrix = absInvViewMatrix * _rootMatrix;

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

void HdGatlingRenderPass::_Execute(const HdRenderPassStateSharedPtr& renderPassState,
                                   const TfTokenVector& renderTags)
{
  TF_UNUSED(renderTags);

  _isConverged = false;

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

  bool sceneChanged = (sceneStateVersion != _lastSceneStateVersion);
  bool renderSettingsChanged = (renderSettingsStateVersion != _lastRenderSettingsVersion);
  bool visibilityChanged = (_lastVisChangeCount != visibilityChangeCount);
  bool aovChanged = (aovId != _lastAovId);

  if (sceneChanged || renderSettingsChanged || visibilityChanged || aovChanged)
  {
    giInvalidateFramebuffer();
  }

  _lastSceneStateVersion = sceneStateVersion;
  _lastSprimIndexVersion = sprimIndexVersion;
  _lastRenderSettingsVersion = renderSettingsStateVersion;
  _lastVisChangeCount = visibilityChangeCount;
  _lastAovId = aovId;

  bool rebuildShaderCache = !_shaderCache || aovChanged || giShaderCacheNeedsRebuild() || renderSettingsChanged;

  bool rebuildGeomCache = !_geomCache || visibilityChanged;

  if (rebuildShaderCache || rebuildGeomCache)
  {
    // Transform scene into camera space to increase floating point precision.
    // FIXME: reintroduce and don't apply rotation
    // https://pharr.org/matt/blog/2018/03/02/rendering-in-camera-space
    //GfMatrix4d viewMatrix = camera->GetTransform().GetInverse();
    _rootMatrix = GfMatrix4d(1.0);// viewMatrix;

    // FIXME: cache results for shader cache rebuild
    std::vector<const GiMaterial*> materials;
    std::vector<const GiMesh*> meshes;
    std::vector<GiMeshInstance> instances;
    _BakeMeshes(renderIndex, _rootMatrix, materials, meshes, instances);

    if (rebuildShaderCache)
    {
      if (_shaderCache)
      {
        giDestroyShaderCache(_shaderCache);
      }

      auto domeLightCameraVisibilityValueIt = _settings.find(HdRenderSettingsTokens->domeLightCameraVisibility);

      GiShaderCacheParams shaderParams = {
        .aovId = aovId,
        .depthOfField = _settings.find(HdGatlingSettingsTokens->depthOfField)->second.Get<bool>(),
        .domeLightCameraVisible = (domeLightCameraVisibilityValueIt == _settings.end()) || domeLightCameraVisibilityValueIt->second.GetWithDefault<bool>(true),
        .filterImportanceSampling = _settings.find(HdGatlingSettingsTokens->filterImportanceSampling)->second.Get<bool>(),
        .materialCount = (uint32_t) materials.size(),
        .materials = materials.data(),
        .nextEventEstimation = _settings.find(HdGatlingSettingsTokens->nextEventEstimation)->second.Get<bool>(),
        .progressiveAccumulation = _settings.find(HdGatlingSettingsTokens->progressiveAccumulation)->second.Get<bool>(),
        .scene = _scene,
        .mediumStackSize = (uint32_t) _settings.find(HdGatlingSettingsTokens->mediumStackSize)->second.Get<int>(),
        .maxVolumeWalkLength = (uint32_t) _settings.find(HdGatlingSettingsTokens->maxVolumeWalkLength)->second.Get<int>()
      };
      _shaderCache = giCreateShaderCache(shaderParams);

      TF_VERIFY(_shaderCache, "Unable to create shader cache");
    }

    if (_shaderCache && (rebuildGeomCache || giGeomCacheNeedsRebuild()))
    {
      if (_geomCache)
      {
        giDestroyGeomCache(_geomCache);
      }

      GiGeomCacheParams geomParams = {
        .meshInstanceCount = (uint32_t) instances.size(),
        .meshInstances = instances.data(),
        .shaderCache = _shaderCache
      };
      _geomCache = giCreateGeomCache(geomParams);

      TF_VERIFY(_geomCache, "Unable to create geom cache");
    }
  }

  if (!_geomCache || !_shaderCache)
  {
    return;
  }

  GfVec4f backgroundColor = aovBinding->clearValue.GetWithDefault<GfVec4f>(GfVec4f(0.f));

  bool clippingEnabled = renderPassState->GetClippingEnabled() &&
                         _settings.find(HdGatlingSettingsTokens->clippingPlanes)->second.Get<bool>();

  GiCameraDesc giCamera;
  _ConstructGiCamera(*camera, giCamera, clippingEnabled);

  GiRenderParams renderParams = {
    .camera = giCamera,
    .geomCache = _geomCache,
    .shaderCache = _shaderCache,
    .renderBuffer = renderBuffer->GetGiRenderBuffer(),
    .lightIntensityMultiplier = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->lightIntensityMultiplier)->second).Get<float>(),
    .maxBounces = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->maxBounces)->second).Get<uint32_t>(),
    .spp = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->spp)->second).Get<uint32_t>(),
    .rrBounceOffset = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->rrBounceOffset)->second).Get<uint32_t>(),
    .rrInvMinTermProb = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->rrInvMinTermProb)->second).Get<float>(),
    .maxSampleValue = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->maxSampleValue)->second).Get<float>(),
    .backgroundColor = { backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3] },
    .domeLight = renderParam->ActiveDomeLight(),
    .scene = _scene
  };

  float* imgData = (float*) renderBuffer->Map();

  GiStatus result = giRender(renderParams, imgData);

  TF_VERIFY(result == GiStatus::Ok, "Unable to render scene.");

  renderBuffer->Unmap();

  _isConverged = true;
}

PXR_NAMESPACE_CLOSE_SCOPE
