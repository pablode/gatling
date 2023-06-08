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
// Sphere Light
//
HdGatlingSphereLight::HdGatlingSphereLight(GiScene* scene, const SdfPath& id)
  : HdLight(id)
  , m_giScene(scene)
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

  // FIXME: intensity, radius (with treatAsPoint)
}

void HdGatlingSphereLight::Finalize(HdRenderParam* renderParam)
{
  giDestroySphereLight(m_giScene, m_giSphereLight);
}

HdDirtyBits HdGatlingSphereLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams | DirtyBits::DirtyTransform;
}

//
// Dome Light
//
HdGatlingDomeLight::HdGatlingDomeLight(GiScene* scene, const SdfPath& id)
  : HdLight(id)
  , m_giScene(scene)
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
    giDestroyDomeLight(m_giScene, m_giDomeLight);
  }
  m_giDomeLight = giCreateDomeLight(m_giScene, path.c_str());

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

  giDestroyDomeLight(m_giScene, m_giDomeLight);
}

HdDirtyBits HdGatlingDomeLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyTransform | DirtyBits::DirtyParams | DirtyBits::DirtyResource;
}

//
// Simple Light
//
HdGatlingSimpleLight::HdGatlingSimpleLight(GiScene* scene, const SdfPath& id)
  : HdLight(id)
  , m_giScene(scene)
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

  // FIXME: implement instantiation & sync logic
}

HdDirtyBits HdGatlingSimpleLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::AllDirty;
}

PXR_NAMESPACE_CLOSE_SCOPE
