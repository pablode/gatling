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

#include <pxr/base/gf/matrix4f.h>
#include <pxr/imaging/glf/simpleLight.h>

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

HdGatlingSphereLight::~HdGatlingSphereLight()
{
  giDestroySphereLight(m_giScene, m_giSphereLight);
}

void HdGatlingSphereLight::Sync(HdSceneDelegate* sceneDelegate,
                           HdRenderParam* renderParam,
                           HdDirtyBits* dirtyBits)
{
  const SdfPath& id = GetId();

  if (*dirtyBits & DirtyBits::DirtyTransform)
  {
    auto T = GfMatrix4f(sceneDelegate->GetTransform(id));

    float transform3x4[3][4] = {
      (float) T[0][0], (float) T[1][0], (float) T[2][0], (float) T[3][0],
      (float) T[0][1], (float) T[1][1], (float) T[2][1], (float) T[3][1],
      (float) T[0][2], (float) T[1][2], (float) T[2][2], (float) T[3][2]
    };
    giSetSphereLightTransform(m_giSphereLight, (float*) transform3x4);
  }

  // TODO: intensity, radius (with treatAsPoint)
}

HdDirtyBits HdGatlingSphereLight::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams | DirtyBits::DirtyTransform;
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
