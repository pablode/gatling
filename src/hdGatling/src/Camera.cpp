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

#include "Camera.h"

#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/camera.h>

#include <cmath>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingCamera::HdGatlingCamera(const SdfPath& id)
  : HdCamera(id)
  , m_vfov(M_PI_2)
{
}

HdGatlingCamera::~HdGatlingCamera()
{
}

float HdGatlingCamera::GetVFov() const
{
  return m_vfov;
}

void HdGatlingCamera::Sync(HdSceneDelegate* sceneDelegate,
                           HdRenderParam* renderParam,
                           HdDirtyBits* dirtyBits)
{
  HdDirtyBits dirtyBitsCopy = *dirtyBits;

  HdCamera::Sync(sceneDelegate, renderParam, &dirtyBitsCopy);

  if (*dirtyBits & DirtyBits::DirtyParams)
  {
    // See https://wiki.panotools.org/Field_of_View
    float aperture = _verticalAperture * GfCamera::APERTURE_UNIT;
    float focalLength = _focalLength * GfCamera::FOCAL_LENGTH_UNIT;
    float vfov = 2.0f * std::atan(aperture / (2.0f * focalLength));

    m_vfov = vfov;
  }

  *dirtyBits = DirtyBits::Clean;
}

HdDirtyBits HdGatlingCamera::GetInitialDirtyBitsMask() const
{
  return DirtyBits::DirtyParams |
         DirtyBits::DirtyTransform;
}

PXR_NAMESPACE_CLOSE_SCOPE
