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
#include "tokens.h"

#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/camera.h>

#include <gtl/gb/Log.h>
#include <gtl/gi/Gi.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
  const static std::unordered_map<TfToken, GiAovId, TfToken::HashFunctor> _aovIdMappings {
    { HdAovTokens->color,                    GiAovId::Color        },
    { HdAovTokens->normal,                   GiAovId::Normal       },
    { HdGatlingAovTokens->debugNee,          GiAovId::NEE          },
    { HdGatlingAovTokens->debugBarycentrics, GiAovId::Barycentrics },
    { HdGatlingAovTokens->debugTexcoords,    GiAovId::Texcoords    },
    { HdGatlingAovTokens->debugBounces,      GiAovId::Bounces      },
    { HdGatlingAovTokens->debugClockCycles,  GiAovId::ClockCycles  },
    { HdGatlingAovTokens->debugOpacity,      GiAovId::Opacity      },
    { HdGatlingAovTokens->debugTangents,     GiAovId::Tangents     },
    { HdGatlingAovTokens->debugBitangents,   GiAovId::Bitangents   },
    { HdGatlingAovTokens->debugThinWalled,   GiAovId::ThinWalled   },
    { HdAovTokens->primId,                   GiAovId::ObjectId     },
    { HdAovTokens->depth,                    GiAovId::Depth        },
    { HdAovTokens->elementId,                GiAovId::FaceId       },
    { HdAovTokens->instanceId,               GiAovId::InstanceId   },
    { HdGatlingAovTokens->debugDoubleSided,  GiAovId::DoubleSided  },
  };

  std::vector<GiAovBinding> _PrepareAovBindings(const HdRenderPassAovBindingVector& aovBindings)
  {
    std::vector<GiAovBinding> result;

    for (const HdRenderPassAovBinding& binding : aovBindings)
    {
      HdGatlingRenderBuffer* renderBuffer = static_cast<HdGatlingRenderBuffer*>(binding.renderBuffer);
      const TfToken& name = binding.aovName;

      auto it = _aovIdMappings.find(name);
      if (it == _aovIdMappings.end())
      {
        TF_RUNTIME_ERROR(TfStringPrintf("Unsupported AOV %s", name.GetText()));
        renderBuffer->SetConverged(true);
        continue;
      }

      auto valueType = HdGetValueTupleType(binding.clearValue).type;
      size_t valueSize = HdDataSizeOfType(valueType);
      const void* valuePtr = HdGetValueData(binding.clearValue);

      GiAovBinding b;
      b.aovId = it->second;
      memcpy(&b.clearValue[0], valuePtr, valueSize);
      b.renderBuffer = renderBuffer->GetGiRenderBuffer();

      result.push_back(b);
    }

    return result;
  }

  bool _IsInteractive(const HdRenderSettingsMap& settings)
  {
    auto settingIt = settings.find(HdRenderSettingsTokens->enableInteractive);
    if (settingIt != settings.end())
    {
      return settingIt->second.Get<bool>();
    }

    // https://www.sidefx.com/docs/hdk/_h_d_k__u_s_d_hydra.html#HDK_USDHydraCustomSettingsInteractive
    const static TfToken houdiniInteractive("houdini:interactive", TfToken::Immortal);
    settingIt = settings.find(houdiniInteractive);

    if (settingIt != settings.end())
    {
      const VtValue& val = settingIt->second;

      const char* str = nullptr;
      if (val.IsHolding<std::string>())
      {
        str = val.UncheckedGet<std::string>().c_str();
      }
      else if (val.IsHolding<TfToken>())
      {
        str = val.UncheckedGet<TfToken>().GetText();
      }
      else
      {
        // FIXME: need to build against Houdini SDK and extract UT_StringHolder in this case
        GB_ERROR("failed to get string from houdini:interactive setting");
        return false;
      }

      return strcmp(str, "normal") != 0;
    }

    return true;
  }
}

HdGatlingRenderPass::HdGatlingRenderPass(HdRenderIndex* index,
                                         const HdRprimCollection& collection,
                                         const HdRenderSettingsMap& settings,
                                         GiScene* scene)
  : HdRenderPass(index, collection)
  , _scene(scene)
  , _settings(settings)
  , _isConverged(false)
{
}

HdGatlingRenderPass::~HdGatlingRenderPass()
{
}

bool HdGatlingRenderPass::IsConverged() const
{
  return _isConverged;
}

void HdGatlingRenderPass::_ConstructGiCamera(const HdCamera& camera, GiCameraDesc& giCamera) const
{
  const GfMatrix4d& transform = camera.GetTransform();

  GfVec3d position = transform.Transform(GfVec3d(0.0, 0.0, 0.0));
  GfVec3d forward = transform.TransformDir(GfVec3d(0.0, 0.0, -1.0));
  GfVec3d up = transform.TransformDir(GfVec3d(0.0, 1.0, 0.0));

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
  giCamera.clipStart = camera.GetClippingRange().GetMin();
  giCamera.clipEnd = camera.GetClippingRange().GetMax();
  giCamera.exposure = camera.GetExposure();
}

void HdGatlingRenderPass::_Execute(const HdRenderPassStateSharedPtr& renderPassState,
                                   const TfTokenVector& renderTags)
{
  TF_UNUSED(renderTags);

  const HdCamera* camera = renderPassState->GetCamera();
  if (!camera)
  {
    return;
  }

  const auto& hdAovBindings = renderPassState->GetAovBindings();

  std::vector<GiAovBinding> aovBindings = _PrepareAovBindings(hdAovBindings);
  if (aovBindings.empty())
  {
    // If this is due to an unsupported AOV, we already logged an error about it.
    return;
  }

  HdRenderIndex* renderIndex = GetRenderIndex();
  HdChangeTracker& changeTracker = renderIndex->GetChangeTracker();
  HdRenderDelegate* renderDelegate = renderIndex->GetRenderDelegate();
  HdGatlingRenderParam* renderParam = static_cast<HdGatlingRenderParam*>(renderDelegate->GetRenderParam());

  bool clippingPlanes = renderPassState->GetClippingEnabled() &&
                        _settings.find(HdGatlingSettingsTokens->clippingPlanes)->second.Get<bool>();

  auto domeLightCameraVisibilityValueIt = _settings.find(HdRenderSettingsTokens->domeLightCameraVisibility);

  GiCameraDesc giCamera;
  _ConstructGiCamera(*camera, giCamera);

  GiRenderParams renderParams = {
    .aovBindings = aovBindings,
    .camera = giCamera,
    .domeLight = renderParam->ActiveDomeLight(),
    .renderSettings = {
      .clippingPlanes = clippingPlanes,
      .depthOfField = _settings.find(HdGatlingSettingsTokens->depthOfField)->second.Get<bool>(),
      .domeLightCameraVisible = (domeLightCameraVisibilityValueIt == _settings.end()) || domeLightCameraVisibilityValueIt->second.GetWithDefault<bool>(true),
      .filterImportanceSampling = _settings.find(HdGatlingSettingsTokens->filterImportanceSampling)->second.Get<bool>(),
      .jitteredSampling = _settings.find(HdGatlingSettingsTokens->jitteredSampling)->second.Get<bool>(),
      .lightIntensityMultiplier = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->lightIntensityMultiplier)->second).Get<float>(),
      .maxBounces = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->maxBounces)->second).Get<uint32_t>(),
      .maxSampleValue = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->maxSampleValue)->second).Get<float>(),
#ifndef HDGATLING_DISABLE_VOLUME_SAMPLING
      .maxVolumeWalkLength = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->maxVolumeWalkLength)->second).Get<uint32_t>(),
      .mediumStackSize = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->mediumStackSize)->second).Get<uint32_t>(),
#endif
      .metersPerSceneUnit = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->stageMetersPerUnit)->second).Get<float>(),
      .nextEventEstimation = _settings.find(HdGatlingSettingsTokens->nextEventEstimation)->second.Get<bool>(),
      .progressiveAccumulation = _settings.find(HdGatlingSettingsTokens->progressiveAccumulation)->second.Get<bool>(),
      .rrBounceOffset = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->rrBounceOffset)->second).Get<uint32_t>(),
      .rrInvMinTermProb = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->rrInvMinTermProb)->second).Get<float>(),
      .spp = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->spp)->second).Get<uint32_t>()
    },
    .scene = _scene
  };

  GiStatus result = giRender(renderParams);

  TF_VERIFY(result == GiStatus::Ok, "Unable to render scene.");

  _isConverged = !_IsInteractive(_settings);

  for (const auto& aovBinding : hdAovBindings)
  {
    auto renderBuffer = static_cast<HdGatlingRenderBuffer*>(aovBinding.renderBuffer);
    renderBuffer->SetConverged(_isConverged);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
