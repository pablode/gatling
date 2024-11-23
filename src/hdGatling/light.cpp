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

#include "light.h"
#include "renderParam.h"

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

#include <gtl/gi/Gi.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
  float _AreaEllipsoid(float radiusX, float radiusY, float radiusZ)
  {
    float ab = powf(radiusX * radiusY, 1.6f);
    float ac = powf(radiusX * radiusZ, 1.6f);
    float bc = powf(radiusY * radiusZ, 1.6f);
    return powf((ab + ac + bc) / 3.0f, 1.0f / 1.6f) * 4.0f * M_PI;
  }
}

//
// Base Light
//

HdGatlingLight::HdGatlingLight(const SdfPath& id, GiScene* scene)
  : HdLight(id)
  , _scene(scene)
{
}

// We strive to conform to following UsdLux-enhancing specification:
// https://github.com/anderslanglands/light_comparison/blob/777ccc7afd1c174a5dcbbde964ced950eb3af11b/specification/specification.md
GfVec3f HdGatlingLight::_CalcBaseEmission(HdSceneDelegate* sceneDelegate, float normalizeFactor = 1.0f)
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

  TF_AXIOM(normalizeFactor > 0.0f);

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
  _giSphereLight = giCreateSphereLight(scene);
}

void HdGatlingSphereLight::Sync(HdSceneDelegate* sceneDelegate,
                                [[maybe_unused]] HdRenderParam* renderParam,
                                HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  const GfMatrix4f transform(sceneDelegate->GetTransform(id));

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    GfVec3f pos = transform.Transform(GfVec3f(0.0f, 0.0f, 0.0f));
    giSetSphereLightPosition(_giSphereLight, pos.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedRadius = sceneDelegate->GetLightParamValue(id, HdLightTokens->radius);
    float radius = boxedRadius.GetWithDefault<float>(0.5f);
    float radiusX = transform.TransformDir(GfVec3f{ radius, 0.0f, 0.0f })[0];
    float radiusY = transform.TransformDir(GfVec3f{ 0.0f, radius, 0.0f })[1];
    float radiusZ = transform.TransformDir(GfVec3f{ 0.0f, 0.0f, radius })[2];

    VtValue boxedNormalize = sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize);
    bool normalize = boxedNormalize.GetWithDefault<bool>(false);
    float area = _AreaEllipsoid(radiusX, radiusY, radiusZ);
    float normalizeFactor = (normalize && area > 0.0f) ? area : 1.0f;
    GfVec3f baseEmission = _CalcBaseEmission(sceneDelegate, normalizeFactor);

    VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
    float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
    VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
    float specular = boxedSpecular.GetWithDefault<float>(1.0f);

    giSetSphereLightRadius(_giSphereLight, radiusX, radiusY, radiusZ);
    giSetSphereLightBaseEmission(_giSphereLight, baseEmission.data());
    giSetSphereLightDiffuseSpecular(_giSphereLight, diffuse, specular);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingSphereLight::Finalize([[maybe_unused]] HdRenderParam* renderParam)
{
  giDestroySphereLight(_scene, _giSphereLight);
}

//
// Distant Light
//
HdGatlingDistantLight::HdGatlingDistantLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
  _giDistantLight = giCreateDistantLight(scene);
}

void HdGatlingDistantLight::Sync(HdSceneDelegate* sceneDelegate,
                                 [[maybe_unused]] HdRenderParam* renderParam,
                                 HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    const GfMatrix4f transform(sceneDelegate->GetTransform(id));
    GfMatrix4f normalMatrix(transform.GetInverse().GetTranspose());

    GfVec3f dir = normalMatrix.TransformDir(GfVec3f(0.0f, 0.0f, -1.0f));
    dir.Normalize();

    giSetDistantLightDirection(_giDistantLight, dir.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedAngle = sceneDelegate->GetLightParamValue(id, HdLightTokens->angle);
    float angle = GfDegreesToRadians(boxedAngle.GetWithDefault<float>(0.53f));

    VtValue boxedNormalize = sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize);
    bool normalize = boxedNormalize.GetWithDefault<bool>(false);
    float sinHalfAngle = sinf(angle * 0.5f);
    float normalizeFactor = (sinHalfAngle > 1.0e-6f && normalize) ? (GfSqr(sinHalfAngle) * M_PI) : 1.0f;
    GfVec3f baseEmission = _CalcBaseEmission(sceneDelegate, normalizeFactor);

    VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
    float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
    VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
    float specular = boxedSpecular.GetWithDefault<float>(1.0f);

    giSetDistantLightAngle(_giDistantLight, angle);
    giSetDistantLightBaseEmission(_giDistantLight, baseEmission.data());
    giSetDistantLightDiffuseSpecular(_giDistantLight, diffuse, specular);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingDistantLight::Finalize([[maybe_unused]] HdRenderParam* renderParam)
{
  giDestroyDistantLight(_scene, _giDistantLight);
}

//
// Rect Light
//
HdGatlingRectLight::HdGatlingRectLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
  _giRectLight = giCreateRectLight(scene);
}

void HdGatlingRectLight::Sync(HdSceneDelegate* sceneDelegate,
                              [[maybe_unused]] HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  const GfMatrix4f transform(sceneDelegate->GetTransform(id));

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    GfVec3f origin = transform.Transform(GfVec3f(0.0f, 0.0f, 0.0f));
    GfVec3f t0 = transform.TransformDir(GfVec3f(1.0f, 0.0f, 0.0f));
    t0.Normalize();
    GfVec3f t1 = transform.TransformDir(GfVec3f(0.0f, 1.0f, 0.0f));
    t1.Normalize();

    giSetRectLightOrigin(_giRectLight, origin.data());
    giSetRectLightTangents(_giRectLight, t0.data(), t1.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedWidth = sceneDelegate->GetLightParamValue(id, HdLightTokens->width);
    float width = boxedWidth.GetWithDefault<float>(1.0f);
    width = transform.TransformDir(GfVec3f{ width, 0.0f, 0.0f })[0];

    VtValue boxedHeight = sceneDelegate->GetLightParamValue(id, HdLightTokens->height);
    float height = boxedHeight.GetWithDefault<float>(1.0f);
    height = transform.TransformDir(GfVec3f{ 0.0f, height, 0.0f })[1];

    VtValue boxedNormalize = sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize);
    bool normalize = boxedNormalize.GetWithDefault<bool>(false);
    float area = width * height;
    float normalizeFactor = (normalize && area > 0.0f) ? area : 1.0f;
    GfVec3f baseEmission = _CalcBaseEmission(sceneDelegate, normalizeFactor);

    VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
    float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
    VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
    float specular = boxedSpecular.GetWithDefault<float>(1.0f);

    giSetRectLightDimensions(_giRectLight, width, height);
    giSetRectLightBaseEmission(_giRectLight, baseEmission.data());
    giSetRectLightDiffuseSpecular(_giRectLight, diffuse, specular);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingRectLight::Finalize([[maybe_unused]] HdRenderParam* renderParam)
{
  giDestroyRectLight(_scene, _giRectLight);
}

//
// Disk Light
//
HdGatlingDiskLight::HdGatlingDiskLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
  _giDiskLight = giCreateDiskLight(scene);
}

void HdGatlingDiskLight::Sync(HdSceneDelegate* sceneDelegate,
                              [[maybe_unused]] HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  const GfMatrix4f transform(sceneDelegate->GetTransform(id));

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    GfVec3f origin = transform.Transform(GfVec3f(0.0f, 0.0f, 0.0f));
    GfVec3f t0 = transform.TransformDir(GfVec3f(1.0f, 0.0f, 0.0f));
    t0.Normalize();
    GfVec3f t1 = transform.TransformDir(GfVec3f(0.0f, 1.0f, 0.0f));
    t1.Normalize();

    giSetDiskLightOrigin(_giDiskLight, origin.data());
    giSetDiskLightTangents(_giDiskLight, t0.data(), t1.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedRadius = sceneDelegate->GetLightParamValue(id, HdLightTokens->radius);
    float radius = boxedRadius.GetWithDefault<float>(0.5f);
    float radiusX = transform.TransformDir(GfVec3f{ radius, 0.0f, 0.0f })[0];
    float radiusY = transform.TransformDir(GfVec3f{ 0.0f, radius, 0.0f })[1];

    VtValue boxedNormalize = sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize);
    bool normalize = boxedNormalize.GetWithDefault<bool>(false);
    float area = radiusX * radiusY * M_PI;
    float normalizeFactor = (normalize && area > 0.0f) ? area : 1.0f;
    GfVec3f baseEmission = _CalcBaseEmission(sceneDelegate, normalizeFactor);

    VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
    float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
    VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
    float specular = boxedSpecular.GetWithDefault<float>(1.0f);

    giSetDiskLightRadius(_giDiskLight, radiusX, radiusY);
    giSetDiskLightBaseEmission(_giDiskLight, baseEmission.data());
    giSetDiskLightDiffuseSpecular(_giDiskLight, diffuse, specular);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingDiskLight::Finalize([[maybe_unused]] HdRenderParam* renderParam)
{
  giDestroyDiskLight(_scene, _giDiskLight);
}

//
// Dome Light
//
HdGatlingDomeLight::HdGatlingDomeLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
}

void HdGatlingDomeLight::Sync(HdSceneDelegate* sceneDelegate,
                              [[maybe_unused]] HdRenderParam* renderParam,
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
    TF_WARN("%s:%s does not hold SdfAssetPath - ignoring!",
      id.GetText(), HdLightTokens->textureFile.GetText());
    return;
  }

  const SdfAssetPath& assetPath = boxedTextureFile.UncheckedGet<SdfAssetPath>();

  std::string path = assetPath.GetResolvedPath();
  if (path.empty())
  {
    TF_RUNTIME_ERROR("Unable to resolve asset path!");
    return;
  }

  if (_giDomeLight)
  {
    DestroyDomeLight(renderParam);
  }

  if (!sceneDelegate->GetVisible(id))
  {
    return;
  }

  // FIXME: don't recreate on transform change
  _giDomeLight = giCreateDomeLight(_scene, path.c_str());

  const GfMatrix4d& transform = sceneDelegate->GetTransform(id);
  auto rotateQuat = GfMatrix4f(transform.GetOrthonormalized()).ExtractRotationQuat();
  float rawQuatData[4] = { rotateQuat.GetImaginary()[0], rotateQuat.GetImaginary()[1], rotateQuat.GetImaginary()[2], -/*flip handedness*/rotateQuat.GetReal() };
  giSetDomeLightRotation(_giDomeLight, rawQuatData);

  GfVec3f baseEmission = _CalcBaseEmission(sceneDelegate);
  giSetDomeLightBaseEmission(_giDomeLight, baseEmission.data());

  VtValue boxedDiffuse = sceneDelegate->GetLightParamValue(id, HdLightTokens->diffuse);
  float diffuse = boxedDiffuse.GetWithDefault<float>(1.0f);
  VtValue boxedSpecular = sceneDelegate->GetLightParamValue(id, HdLightTokens->specular);
  float specular = boxedSpecular.GetWithDefault<float>(1.0f);
  giSetDomeLightDiffuseSpecular(_giDomeLight, diffuse, specular);

  // We need to ensure that the correct dome light is displayed when usdview's additional
  // one has been enabled. Although the type isn't 'simpleLight' (which may be a bug), we
  // can identify usdview's dome light by the GlfSimpleLight data payload it carries.
  bool isOverrideDomeLight = !sceneDelegate->Get(id, HdLightTokens->params).IsEmpty();

  auto rp = static_cast<HdGatlingRenderParam*>(renderParam);
  if (isOverrideDomeLight)
  {
    rp->SetDomeLightOverride(_giDomeLight);
  }
  else
  {
    rp->AddDomeLight(_giDomeLight);
  }
}

void HdGatlingDomeLight::Finalize([[maybe_unused]] HdRenderParam* renderParam)
{
  DestroyDomeLight(renderParam);
}

HdDirtyBits HdGatlingDomeLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyTransform | DirtyBits::DirtyParams | DirtyBits::DirtyResource;
}

void HdGatlingDomeLight::DestroyDomeLight(HdRenderParam* renderParam)
{
  if (!_giDomeLight)
  {
    return;
  }

  auto rp = static_cast<HdGatlingRenderParam*>(renderParam);
  rp->RemoveDomeLight(_giDomeLight);

  giDestroyDomeLight(_giDomeLight);

  _giDomeLight = nullptr;
}

//
// Simple Light
//
HdGatlingSimpleLight::HdGatlingSimpleLight(const SdfPath& id, GiScene* scene)
  : HdGatlingLight(id, scene)
{
}

void HdGatlingSimpleLight::Sync(HdSceneDelegate* sceneDelegate,
                                [[maybe_unused]] HdRenderParam* renderParam,
                                HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  VtValue boxedGlfLight = sceneDelegate->Get(id, HdLightTokens->params);
  if (!boxedGlfLight.IsHolding<GlfSimpleLight>())
  {
    TF_WARN("SimpleLight %s has no data payload - ignoring", id.GetText());
    return;
  }

  const auto& glfLight = boxedGlfLight.UncheckedGet<GlfSimpleLight>();

  if (!glfLight.IsDomeLight())
  {
    if (!_giSphereLight)
    {
      _giSphereLight = giCreateSphereLight(_scene);
    }

    if (*dirtyBits & DirtyBits::DirtyTransform)
    {
      GfVec4f pos = glfLight.GetPosition();
      giSetSphereLightPosition(_giSphereLight, pos.data());
    }

    if (*dirtyBits & DirtyBits::DirtyParams)
    {
      GfVec3f baseEmission = _CalcBaseEmission(sceneDelegate);
      giSetSphereLightBaseEmission(_giSphereLight, baseEmission.data());
    }
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingSimpleLight::Finalize([[maybe_unused]] HdRenderParam* renderParam)
{
  if (_giSphereLight)
  {
    giDestroySphereLight(_scene, _giSphereLight);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
