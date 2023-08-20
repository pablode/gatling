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
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/glf/simpleLight.h>
#include <pxr/usd/sdf/assetPath.h>

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

GfVec3f HdGatlingLight::CalcBaseEmission(HdSceneDelegate* sceneDelegate, float normalizeFactor)
{
  // TODO
  return {};
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
    VtValue boxedIntensity = sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity);
    float intensity = boxedIntensity.Cast<float>().Get<float>();
    giSetSphereLightIntensity(m_giSphereLight, intensity);

    VtValue boxedColor = sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
    GfVec3f color = boxedColor.Cast<GfVec3f>().Get<GfVec3f>();
    giSetSphereLightColor(m_giSphereLight, color.data());

    VtValue boxedRadius = sceneDelegate->GetLightParamValue(id, HdLightTokens->radius);
    float radius = boxedRadius.Cast<float>().Get<float>();
    giSetSphereLightRadius(m_giSphereLight, radius);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingSphereLight::Finalize(HdRenderParam* renderParam)
{
  giDestroySphereLight(m_scene, m_giSphereLight);
}

HdDirtyBits HdGatlingSphereLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams | DirtyBits::DirtyTransform;
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
    VtValue boxedIntensity = sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity);
    float intensity = boxedIntensity.Cast<float>().Get<float>();
    giSetDistantLightIntensity(m_giDistantLight, intensity);

    VtValue boxedColor = sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
    GfVec3f color = boxedColor.Cast<GfVec3f>().Get<GfVec3f>();
    giSetDistantLightColor(m_giDistantLight, color.data());

    VtValue boxedAngle = sceneDelegate->GetLightParamValue(id, HdLightTokens->angle);
    float angle = GfDegreesToRadians(boxedAngle.Cast<float>().Get<float>());
    giSetDistantLightAngle(m_giDistantLight, angle);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingDistantLight::Finalize(HdRenderParam* renderParam)
{
  giDestroyDistantLight(m_scene, m_giDistantLight);
}

HdDirtyBits HdGatlingDistantLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams | DirtyBits::DirtyTransform;
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
    giSetRectLightOrigin(m_giRectLight, origin.data());

    auto dir = sceneDelegate->GetTransform(id).TransformDir(GfVec3f(0.0f, 0.0f, -1.0f));
    giSetRectLightDirection(m_giRectLight, dir.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    VtValue boxedIntensity = sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity);
    float intensity = boxedIntensity.Cast<float>().Get<float>();
    giSetRectLightIntensity(m_giRectLight, intensity);

    VtValue boxedColor = sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
    GfVec3f color = boxedColor.Cast<GfVec3f>().Get<GfVec3f>();
    giSetRectLightColor(m_giRectLight, color.data());

    VtValue boxedWidth = sceneDelegate->GetLightParamValue(id, HdLightTokens->width);
    float width = boxedWidth.Cast<float>().Get<float>();
    VtValue boxedHeight = sceneDelegate->GetLightParamValue(id, HdLightTokens->height);
    float height = boxedHeight.Cast<float>().Get<float>();
    giSetRectLightDimensions(m_giRectLight, width, height);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdGatlingRectLight::Finalize(HdRenderParam* renderParam)
{
  giDestroyRectLight(m_scene, m_giRectLight);
}

HdDirtyBits HdGatlingRectLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams | DirtyBits::DirtyTransform;
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
  TF_UNUSED(renderParam);

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
    // FIXME: don't recreate on transform change
    giDestroyDomeLight(m_scene, m_giDomeLight);
  }
  m_giDomeLight = giCreateDomeLight(m_scene, path.c_str());

  const GfMatrix4d& transform = sceneDelegate->GetTransform(id);
  auto rotateTransform = GfMatrix3f(transform.ExtractRotationMatrix());
  giSetDomeLightTransform(m_giDomeLight, rotateTransform.data());

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
  if (!m_giDomeLight)
  {
    return;
  }

  auto rp = static_cast<HdGatlingRenderParam*>(renderParam);
  rp->RemoveDomeLight(m_giDomeLight);

  giDestroyDomeLight(m_scene, m_giDomeLight);
}

HdDirtyBits HdGatlingDomeLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyTransform | DirtyBits::DirtyParams | DirtyBits::DirtyResource;
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

  if (!glfLight.IsDomeLight() && !m_giSphereLight)
  {
    m_giSphereLight = giCreateSphereLight(m_scene);
  }

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    auto pos = glfLight.GetPosition();
    giSetSphereLightPosition(m_giSphereLight, pos.data());
  }

  if (*dirtyBits & DirtyBits::DirtyParams && glfLight.HasIntensity())
  {
    VtValue boxedIntensity = sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity);
    float intensity = boxedIntensity.Cast<float>().Get<float>();
    giSetSphereLightIntensity(m_giSphereLight, intensity);

    VtValue boxedColor = sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
    GfVec3f color = boxedColor.Cast<GfVec3f>().Get<GfVec3f>();
    giSetSphereLightColor(m_giSphereLight, color.data());
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

HdDirtyBits HdGatlingSimpleLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::AllDirty;
}

PXR_NAMESPACE_CLOSE_SCOPE
