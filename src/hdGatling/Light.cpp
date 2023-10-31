//
// Copyright (C) 2023 Pablo Delgado Krämer
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

#include "Light.h"
#include "RenderParam.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix3f.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/glf/simpleLight.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdLux/blackbody.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

//
// Base Light
//

HdGatlingLight::HdGatlingLight(const SdfPath& id, GiScene* scene)
  : HdLight(id)
  , m_scene(scene)
{
}

// We strive to conform to following UsdLux-enhancing specification:
// https://github.com/anderslanglands/light_comparison/blob/777ccc7afd1c174a5dcbbde964ced950eb3af11b/specification/specification.md
GfVec3f HdGatlingLight::CalcBaseEmission(HdSceneDelegate* sceneDelegate, float normalizeFactor = 1.0f)
{
  const SdfPath& id = GetId();

  VtValue boxedIntensity = sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity);
  float intensity = boxedIntensity.GetWithDefault<float>(1.0f);

  VtValue boxedColor = sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
  GfVec3f color = boxedColor.GetWithDefault<GfVec3f>({1.0f, 1.0f, 1.0f});

  VtValue boxedEnableColorTemperature = sceneDelegate->GetLightParamValue(id, HdLightTokens->enableColorTemperature);
  bool enableColorTemperature = boxedEnableColorTemperature.GetWithDefault<bool>(false);

  VtValue boxedColorTemperature = sceneDelegate->GetLightParamValue(id, HdLightTokens->colorTemperature);
  float colorTemperature = boxedColorTemperature.GetWithDefault<float>(6500.0f);

  VtValue boxedExposureAttr = sceneDelegate->GetLightParamValue(id, HdLightTokens->exposure);
  float exposure = boxedExposureAttr.GetWithDefault<float>(0.0f);

  assert(normalizeFactor > 0.0f);

  float normalizedIntensity = intensity * powf(2.0f, exposure) / normalizeFactor;

  GfVec3f baseEmission = color * normalizedIntensity;

  if (enableColorTemperature)
  {
    baseEmission = GfCompMult(baseEmission, UsdLuxBlackbodyTemperatureAsRgb(colorTemperature));
  }

  return baseEmission;
}

HdDirtyBits HdGatlingLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams | DirtyBits::DirtyTransform;
}

//
// Sphere Light
//
HdGatlingSphereLight::HdGatlingSphereLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
  m_giSphereLight = giCreateSphereLight(scene);
}

void HdGatlingSphereLight::Sync(HdSceneDelegate* sceneDelegate,
                                HdRenderParam* renderParam,
                                HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    auto pos = sceneDelegate->GetTransform(id).Transform(GfVec3f(0.0f, 0.0f, 0.0f));
    giSetSphereLightPosition(m_giSphereLight, pos.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedRadius = sceneDelegate->GetLightParamValue(id, HdLightTokens->radius);
    float radius = boxedRadius.GetWithDefault<float>(0.5f);

    VtValue boxedNormalize = sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize);
    bool normalize = boxedNormalize.GetWithDefault<bool>(false);
    float normalizeFactor = normalize ? ((radius > 1.0e-6f ? 4.0 : 1.0f) * GfSqr(radius) * M_PI) : 1.0f;
    GfVec3f baseEmission = CalcBaseEmission(sceneDelegate, normalizeFactor);

    VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
    float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
    VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
    float specular = boxedSpecular.GetWithDefault<float>(1.0f);

    giSetSphereLightRadius(m_giSphereLight, radius);
    giSetSphereLightBaseEmission(m_giSphereLight, baseEmission.data());
    giSetSphereLightDiffuseSpecular(m_giSphereLight, diffuse, specular);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingSphereLight::Finalize(HdRenderParam* renderParam)
{
  giDestroySphereLight(m_scene, m_giSphereLight);
}

//
// Distant Light
//
HdGatlingDistantLight::HdGatlingDistantLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
  m_giDistantLight = giCreateDistantLight(scene);
}

void HdGatlingDistantLight::Sync(HdSceneDelegate* sceneDelegate,
                                 HdRenderParam* renderParam,
                                 HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    auto dir = sceneDelegate->GetTransform(id).TransformDir(GfVec3f(0.0f, 0.0f, -1.0f));
    giSetDistantLightDirection(m_giDistantLight, dir.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedAngle = sceneDelegate->GetLightParamValue(id, HdLightTokens->angle);
    float angle = GfDegreesToRadians(boxedAngle.GetWithDefault<float>(0.53f));

    VtValue boxedNormalize = sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize);
    bool normalize = boxedNormalize.GetWithDefault<bool>(false);
    float sinHalfAngle = sinf(angle * 0.5f);
    float normalizeFactor = (sinHalfAngle > 1.0e-6f && normalize) ? (GfSqr(sinHalfAngle) * M_PI) : 1.0f;
    GfVec3f baseEmission = CalcBaseEmission(sceneDelegate, normalizeFactor);

    VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
    float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
    VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
    float specular = boxedSpecular.GetWithDefault<float>(1.0f);

    giSetDistantLightAngle(m_giDistantLight, angle);
    giSetDistantLightBaseEmission(m_giDistantLight, baseEmission.data());
    giSetDistantLightDiffuseSpecular(m_giDistantLight, diffuse, specular);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingDistantLight::Finalize(HdRenderParam* renderParam)
{
  giDestroyDistantLight(m_scene, m_giDistantLight);
}

//
// Rect Light
//
HdGatlingRectLight::HdGatlingRectLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
  m_giRectLight = giCreateRectLight(scene);
}

void HdGatlingRectLight::Sync(HdSceneDelegate* sceneDelegate,
                              HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    auto origin = sceneDelegate->GetTransform(id).Transform(GfVec3f(0.0f, 0.0f, 0.0f));
    auto dir = sceneDelegate->GetTransform(id).TransformDir(GfVec3f(0.0f, 0.0f, -1.0f));

    giSetRectLightOrigin(m_giRectLight, origin.data());
    giSetRectLightDirection(m_giRectLight, dir.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedWidth = sceneDelegate->GetLightParamValue(id, HdLightTokens->width);
    float width = boxedWidth.GetWithDefault<float>(1.0f);
    VtValue boxedHeight = sceneDelegate->GetLightParamValue(id, HdLightTokens->height);
    float height = boxedHeight.GetWithDefault<float>(1.0f);

    VtValue boxedNormalize = sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize);
    bool normalize = boxedNormalize.GetWithDefault<bool>(false);
    float area = width * height;
    float normalizeFactor = (area > 0.0f && normalize) ? area : 1.0f;
    GfVec3f baseEmission = CalcBaseEmission(sceneDelegate, normalizeFactor);

    VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
    float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
    VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
    float specular = boxedSpecular.GetWithDefault<float>(1.0f);

    giSetRectLightDimensions(m_giRectLight, width, height);
    giSetRectLightBaseEmission(m_giRectLight, baseEmission.data());
    giSetRectLightDiffuseSpecular(m_giRectLight, diffuse, specular);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingRectLight::Finalize(HdRenderParam* renderParam)
{
  giDestroyRectLight(m_scene, m_giRectLight);
}

//
// Disk Light
//
HdGatlingDiskLight::HdGatlingDiskLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
  m_giDiskLight = giCreateDiskLight(scene);
}

void HdGatlingDiskLight::Sync(HdSceneDelegate* sceneDelegate,
                              HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    auto origin = sceneDelegate->GetTransform(id).Transform(GfVec3f(0.0f, 0.0f, 0.0f));
    auto dir = sceneDelegate->GetTransform(id).TransformDir(GfVec3f(0.0f, 0.0f, -1.0f));

    giSetDiskLightOrigin(m_giDiskLight, origin.data());
    giSetDiskLightDirection(m_giDiskLight, dir.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedRadius = sceneDelegate->GetLightParamValue(id, HdLightTokens->radius);
    float radius = boxedRadius.GetWithDefault<float>(0.5f);

    VtValue boxedNormalize = sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize);
    bool normalize = boxedNormalize.GetWithDefault<bool>(false);
    float normalizeFactor = (radius > 1.0e-6f && normalize) ? (GfSqr(radius) * M_PI) : 1.0f;
    GfVec3f baseEmission = CalcBaseEmission(sceneDelegate, normalizeFactor);

    VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
    float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
    VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
    float specular = boxedSpecular.GetWithDefault<float>(1.0f);

    giSetDiskLightRadius(m_giDiskLight, radius);
    giSetDiskLightBaseEmission(m_giDiskLight, baseEmission.data());
    giSetDiskLightDiffuseSpecular(m_giDiskLight, diffuse, specular);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingDiskLight::Finalize(HdRenderParam* renderParam)
{
  giDestroyDiskLight(m_scene, m_giDiskLight);
}

//
// Dome Light
//
HdGatlingDomeLight::HdGatlingDomeLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
}

void HdGatlingDomeLight::Sync(HdSceneDelegate* sceneDelegate,
                              HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits)
{
  if (!HdChangeTracker::IsDirty(*dirtyBits))
  {
    return;
  }

  *dirtyBits = DirtyBits::Clean;

  const SdfPath& id = GetId();
  VtValue boxedTextureFile = sceneDelegate->GetLightParamValue(id, HdLightTokens->textureFile);
  if (boxedTextureFile.IsEmpty())
  {
    // Hydra runtime warns of empty path; we don't need to repeat it.
    return;
  }

  if (!boxedTextureFile.IsHolding<SdfAssetPath>())
  {
    TF_CODING_ERROR("Param %s does not hold SdfAssetPath - unsupported!", id.GetString().c_str());
    return;
  }

  const SdfAssetPath& assetPath = boxedTextureFile.UncheckedGet<SdfAssetPath>();

  std::string path = assetPath.GetResolvedPath();
  if (path.empty())
  {
    TF_CODING_ERROR("Asset path is not resolved!");
    return;
  }

  if (m_giDomeLight)
  {
    DestroyDomeLight(renderParam);
  }

  if (!sceneDelegate->GetVisible(id))
  {
    return;
  }

  // FIXME: don't recreate on transform change
  m_giDomeLight = giCreateDomeLight(m_scene, path.c_str());

  const GfMatrix4d& transform = sceneDelegate->GetTransform(id);
  auto rotateQuat = GfMatrix4f(transform.GetOrthonormalized()).ExtractRotationQuat();
  float rawQuatData[4] = { rotateQuat.GetImaginary()[0], rotateQuat.GetImaginary()[1], rotateQuat.GetImaginary()[2], -/*flip handedness*/rotateQuat.GetReal() };
  giSetDomeLightRotation(m_giDomeLight, rawQuatData);

  GfVec3f baseEmission = CalcBaseEmission(sceneDelegate);
  giSetDomeLightBaseEmission(m_giDomeLight, baseEmission.data());

  VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
  float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
  VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
  float specular = boxedSpecular.GetWithDefault<float>(1.0f);
  giSetDomeLightDiffuseSpecular(m_giDomeLight, diffuse, specular);

  // We need to ensure that the correct dome light is displayed when usdview's additional
  // one has been enabled. Although the type isn't 'simpleLight' (which may be a bug), we
  // can identify usdview's dome light by the GlfSimpleLight data payload it carries.
  bool isOverrideDomeLight = !sceneDelegate->Get(id, HdLightTokens->params).IsEmpty();

  auto rp = static_cast<HdGatlingRenderParam*>(renderParam);
  if (isOverrideDomeLight)
  {
    rp->SetDomeLightOverride(m_giDomeLight);
  }
  else
  {
    rp->AddDomeLight(m_giDomeLight);
  }
}

void HdGatlingDomeLight::Finalize(HdRenderParam* renderParam)
{
  DestroyDomeLight(renderParam);
}

HdDirtyBits HdGatlingDomeLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyTransform | DirtyBits::DirtyParams | DirtyBits::DirtyResource;
}

void HdGatlingDomeLight::DestroyDomeLight(HdRenderParam* renderParam)
{
  if (!m_giDomeLight)
  {
    return;
  }

  auto rp = static_cast<HdGatlingRenderParam*>(renderParam);
  rp->RemoveDomeLight(m_giDomeLight);

  giDestroyDomeLight(m_scene, m_giDomeLight);

  m_giDomeLight = nullptr;
}

//
// Simple Light
//
HdGatlingSimpleLight::HdGatlingSimpleLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
}

void HdGatlingSimpleLight::Sync(HdSceneDelegate* sceneDelegate,
                                HdRenderParam* renderParam,
                                HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  VtValue boxedGlfLight = sceneDelegate->Get(id, HdLightTokens->params);
  if (!boxedGlfLight.IsHolding<GlfSimpleLight>())
  {
    TF_CODING_ERROR("SimpleLight has no data payload!");
    return;
  }

  const auto& glfLight = boxedGlfLight.UncheckedGet<GlfSimpleLight>();

  if (!glfLight.IsDomeLight())
  {
    if (!m_giSphereLight)
    {
      m_giSphereLight = giCreateSphereLight(m_scene);
    }

    if (*dirtyBits & DirtyBits::DirtyTransform)
    {
      auto pos = glfLight.GetPosition();
      giSetSphereLightPosition(m_giSphereLight, pos.data());
    }

    if (*dirtyBits & DirtyBits::DirtyParams)
    {
      GfVec3f baseEmission = CalcBaseEmission(sceneDelegate);
      giSetSphereLightBaseEmission(m_giSphereLight, baseEmission.data());
    }
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingSimpleLight::Finalize(HdRenderParam* renderParam)
{
  if (m_giSphereLight)
  {
    giDestroySphereLight(m_scene, m_giSphereLight);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
