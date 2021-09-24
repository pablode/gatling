#include "RenderDelegate.h"
#include "RenderPass.h"
#include "Camera.h"
#include "Mesh.h"
#include "Instancer.h"
#include "RenderBuffer.h"
#include "Material.h"
#include "Tokens.h"

#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/base/gf/vec4f.h>

#include <memory>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingRenderDelegate::HdGatlingRenderDelegate(const HdRenderSettingsMap& settingsMap)
{
  m_resourceRegistry = std::make_shared<HdResourceRegistry>();

  m_settingDescriptors.push_back(HdRenderSettingDescriptor{ "Samples per pixel", HdGatlingSettingsTokens->spp, VtValue{8} });
  m_settingDescriptors.push_back(HdRenderSettingDescriptor{ "Max bounces", HdGatlingSettingsTokens->max_bounces, VtValue{4} });
  m_settingDescriptors.push_back(HdRenderSettingDescriptor{ "Russian roulette bounce offset", HdGatlingSettingsTokens->rr_bounce_offset, VtValue{2} });
  m_settingDescriptors.push_back(HdRenderSettingDescriptor{ "Russian roulette inverse minimum terminate probability", HdGatlingSettingsTokens->rr_inv_min_term_prob, VtValue{1.0f} });

  _PopulateDefaultSettings(m_settingDescriptors);

  for (const auto& setting : settingsMap)
  {
    const TfToken& key = setting.first;
    const VtValue& value = setting.second;

    _settingsMap[key] = value;
  }
}

HdGatlingRenderDelegate::~HdGatlingRenderDelegate()
{
}

HdRenderSettingDescriptorList HdGatlingRenderDelegate::GetRenderSettingDescriptors() const
{
  return m_settingDescriptors;
}

HdRenderPassSharedPtr HdGatlingRenderDelegate::CreateRenderPass(HdRenderIndex* index,
                                                                const HdRprimCollection& collection)
{
  return HdRenderPassSharedPtr(new HdGatlingRenderPass(index, collection, _settingsMap));
}

HdResourceRegistrySharedPtr HdGatlingRenderDelegate::GetResourceRegistry() const
{
  return m_resourceRegistry;
}

void HdGatlingRenderDelegate::CommitResources(HdChangeTracker* tracker)
{
  TF_UNUSED(tracker);

  // We delay BVH building and GPU uploads to the next render call.
}

HdInstancer* HdGatlingRenderDelegate::CreateInstancer(HdSceneDelegate* delegate,
                                                      const SdfPath& id)
{
  return new HdGatlingInstancer(delegate, id);
}

void HdGatlingRenderDelegate::DestroyInstancer(HdInstancer* instancer)
{
  delete instancer;
}

HdAovDescriptor HdGatlingRenderDelegate::GetDefaultAovDescriptor(const TfToken& name) const
{
  TF_UNUSED(name);

  HdAovDescriptor aovDescriptor;
  aovDescriptor.format = HdFormatFloat32Vec4;
  aovDescriptor.multiSampled = false;
  aovDescriptor.clearValue = GfVec4f(0.0f, 0.0f, 0.0f, 0.0f);
  return aovDescriptor;
}

const TfTokenVector SUPPORTED_RPRIM_TYPES =
{
  HdPrimTypeTokens->mesh
};

const TfTokenVector& HdGatlingRenderDelegate::GetSupportedRprimTypes() const
{
  return SUPPORTED_RPRIM_TYPES;
}

HdRprim* HdGatlingRenderDelegate::CreateRprim(const TfToken& typeId,
                                              const SdfPath& rprimId)
{
  if (typeId == HdPrimTypeTokens->mesh)
  {
    return new HdGatlingMesh(rprimId);
  }

  return nullptr;
}

void HdGatlingRenderDelegate::DestroyRprim(HdRprim* rprim)
{
  delete rprim;
}

const TfTokenVector SUPPORTED_SPRIM_TYPES =
{
  HdPrimTypeTokens->camera,
  HdPrimTypeTokens->material
};

const TfTokenVector& HdGatlingRenderDelegate::GetSupportedSprimTypes() const
{
  return SUPPORTED_SPRIM_TYPES;
}

HdSprim* HdGatlingRenderDelegate::CreateSprim(const TfToken& typeId,
                                              const SdfPath& sprimId)
{
  if (typeId == HdPrimTypeTokens->camera)
  {
    return new HdGatlingCamera(sprimId);
  }
  else if (typeId == HdPrimTypeTokens->material)
  {
    return new HdGatlingMaterial(sprimId);
  }

  return nullptr;
}

HdSprim* HdGatlingRenderDelegate::CreateFallbackSprim(const TfToken& typeId)
{
  const SdfPath& sprimId = SdfPath::EmptyPath();

  return CreateSprim(typeId, sprimId);
}

void HdGatlingRenderDelegate::DestroySprim(HdSprim* sprim)
{
  delete sprim;
}

const TfTokenVector SUPPORTED_BPRIM_TYPES =
{
  HdPrimTypeTokens->renderBuffer
};

const TfTokenVector& HdGatlingRenderDelegate::GetSupportedBprimTypes() const
{
  return SUPPORTED_BPRIM_TYPES;
}

HdBprim* HdGatlingRenderDelegate::CreateBprim(const TfToken& typeId,
                                              const SdfPath& bprimId)
{
  if (typeId == HdPrimTypeTokens->renderBuffer)
  {
    return new HdGatlingRenderBuffer(bprimId);
  }

  return nullptr;
}

HdBprim* HdGatlingRenderDelegate::CreateFallbackBprim(const TfToken& typeId)
{
  const SdfPath& bprimId = SdfPath::EmptyPath();

  return CreateBprim(typeId, bprimId);
}

void HdGatlingRenderDelegate::DestroyBprim(HdBprim* bprim)
{
  delete bprim;
}

PXR_NAMESPACE_CLOSE_SCOPE
