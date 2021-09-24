#pragma once

#include <pxr/imaging/hd/renderDelegate.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingRenderDelegate final : public HdRenderDelegate
{
public:
  HdGatlingRenderDelegate(const HdRenderSettingsMap& settingsMap);

  ~HdGatlingRenderDelegate() override;

public:
  HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;

public:
  HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index,
                                         const HdRprimCollection& collection) override;

  HdResourceRegistrySharedPtr GetResourceRegistry() const override;

  void CommitResources(HdChangeTracker* tracker) override;

  HdInstancer* CreateInstancer(HdSceneDelegate* delegate,
                               const SdfPath& id) override;

  void DestroyInstancer(HdInstancer* instancer) override;

  HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const override;

public:
  /* Rprim */
  const TfTokenVector& GetSupportedRprimTypes() const override;

  HdRprim* CreateRprim(const TfToken& typeId, const SdfPath& rprimId) override;

  void DestroyRprim(HdRprim* rPrim) override;

  /* Sprim */
  const TfTokenVector& GetSupportedSprimTypes() const override;

  HdSprim* CreateSprim(const TfToken& typeId, const SdfPath& sprimId) override;

  HdSprim* CreateFallbackSprim(const TfToken& typeId) override;

  void DestroySprim(HdSprim* sprim) override;

  /* Bprim */
  const TfTokenVector& GetSupportedBprimTypes() const override;

  HdBprim* CreateBprim(const TfToken& typeId, const SdfPath& bprimId) override;

  HdBprim* CreateFallbackBprim(const TfToken& typeId) override;

  void DestroyBprim(HdBprim* bprim) override;

private:
  HdRenderSettingDescriptorList m_settingDescriptors;
  HdResourceRegistrySharedPtr m_resourceRegistry;
};

PXR_NAMESPACE_CLOSE_SCOPE
