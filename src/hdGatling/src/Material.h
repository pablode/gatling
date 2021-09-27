#pragma once

#include <pxr/imaging/hd/material.h>

#include <gi.h>

#include "MaterialNetworkTranslator.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingMaterial final : public HdMaterial
{
public:
  HdGatlingMaterial(const SdfPath& id,
                    const MaterialNetworkTranslator& translator);

  ~HdGatlingMaterial() override;

public:
  HdDirtyBits GetInitialDirtyBitsMask() const override;

  void Sync(HdSceneDelegate* sceneDelegate,
            HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits) override;

public:
  const gi_material* GetGiMaterial() const;

private:
  const MaterialNetworkTranslator& m_translator;
  gi_material* m_giMaterial = nullptr;
};

PXR_NAMESPACE_CLOSE_SCOPE
