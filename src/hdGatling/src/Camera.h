#pragma once

#include <pxr/imaging/hd/camera.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingCamera final : public HdCamera
{
public:
  HdGatlingCamera(const SdfPath& id);

  ~HdGatlingCamera() override;

public:
  float GetVFov() const;

public:
  void Sync(HdSceneDelegate* sceneDelegate,
            HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
  float m_vfov;
};

PXR_NAMESPACE_CLOSE_SCOPE
