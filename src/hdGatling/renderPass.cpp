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

#include <gtl/gi/Gi.h>

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
    { HdGatlingAovTokens->debugThinWalled,   GiAovId::ThinWalled   },
#endif
    { HdAovTokens->primId,                   GiAovId::ObjectId     },
  };

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

void HdGatlingRenderPass::_ConstructGiCamera(const HdCamera& camera, GiCameraDesc& giCamera, bool clippingEnabled) const
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

  GiAovId aovId = _GetAovId(aovBinding->aovName);

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

  GfVec4f backgroundColor = aovBinding->clearValue.GetWithDefault<GfVec4f>(GfVec4f(0.f));

  bool clippingEnabled = renderPassState->GetClippingEnabled() &&
                         _settings.find(HdGatlingSettingsTokens->clippingPlanes)->second.Get<bool>();

  auto domeLightCameraVisibilityValueIt = _settings.find(HdRenderSettingsTokens->domeLightCameraVisibility);

  GiCameraDesc giCamera;
  _ConstructGiCamera(*camera, giCamera, clippingEnabled);

  GiRenderParams renderParams = {
    .aovId = aovId,
    .camera = giCamera,
    .domeLight = renderParam->ActiveDomeLight(),
    .renderBuffer = renderBuffer->GetGiRenderBuffer(),
    .renderSettings = {
      .backgroundColor = { backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3] },
      .depthOfField = _settings.find(HdGatlingSettingsTokens->depthOfField)->second.Get<bool>(),
      .domeLightCameraVisible = (domeLightCameraVisibilityValueIt == _settings.end()) || domeLightCameraVisibilityValueIt->second.GetWithDefault<bool>(true),
      .filterImportanceSampling = _settings.find(HdGatlingSettingsTokens->filterImportanceSampling)->second.Get<bool>(),
      .lightIntensityMultiplier = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->lightIntensityMultiplier)->second).Get<float>(),
      .maxBounces = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->maxBounces)->second).Get<uint32_t>(),
      .maxSampleValue = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->maxSampleValue)->second).Get<float>(),
      .maxVolumeWalkLength = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->maxVolumeWalkLength)->second).Get<uint32_t>(),
      .mediumStackSize = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->mediumStackSize)->second).Get<uint32_t>(),
      .nextEventEstimation = _settings.find(HdGatlingSettingsTokens->nextEventEstimation)->second.Get<bool>(),
      .progressiveAccumulation = _settings.find(HdGatlingSettingsTokens->progressiveAccumulation)->second.Get<bool>(),
      .rrBounceOffset = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->rrBounceOffset)->second).Get<uint32_t>(),
      .rrInvMinTermProb = VtValue::Cast<float>(_settings.find(HdGatlingSettingsTokens->rrInvMinTermProb)->second).Get<float>(),
      .spp = VtValue::Cast<uint32_t>(_settings.find(HdGatlingSettingsTokens->spp)->second).Get<uint32_t>()
    },
    .scene = _scene
  };

  float* imgData = (float*) renderBuffer->Map();

  GiStatus result = giRender(renderParams, imgData);

  TF_VERIFY(result == GiStatus::Ok, "Unable to render scene.");

  renderBuffer->Unmap();

  _isConverged = true;
}

PXR_NAMESPACE_CLOSE_SCOPE
