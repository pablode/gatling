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
