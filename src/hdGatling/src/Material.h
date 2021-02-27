#pragma once

#include <pxr/imaging/hd/material.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingMaterial final : public HdMaterial
{
public:
  HdGatlingMaterial(const SdfPath& id);

  ~HdGatlingMaterial() override;

public:
  const gi_material& GetGiMaterial() const;

public:
  void Sync(HdSceneDelegate* sceneDelegate,
            HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
  void _ReadMaterialNetwork(const HdMaterialNetwork* network);

private:
  gi_material m_material;
};

PXR_NAMESPACE_CLOSE_SCOPE
