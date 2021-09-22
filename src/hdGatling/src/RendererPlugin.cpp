#include "RendererPlugin.h"
#include "RenderDelegate.h"

#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/base/plug/plugin.h>
#include "pxr/base/plug/thisPlugin.h"

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfType)
{
  HdRendererPluginRegistry::Define<HdGatlingRendererPlugin>();
}

HdGatlingRendererPlugin::HdGatlingRendererPlugin()
{
  PlugPluginPtr plugin = PLUG_THIS_PLUGIN;
  const char* resourcePath = plugin->GetResourcePath().c_str();
  printf("Resource path: %s\n", resourcePath);

  int initResult = giInitialize(resourcePath);

  m_isSupported = (initResult == GI_OK);
}

HdGatlingRendererPlugin::~HdGatlingRendererPlugin()
{
  giTerminate();
}

HdRenderDelegate* HdGatlingRendererPlugin::CreateRenderDelegate()
{
  HdRenderSettingsMap settingsMap;

  return new HdGatlingRenderDelegate(settingsMap);
}

HdRenderDelegate* HdGatlingRendererPlugin::CreateRenderDelegate(const HdRenderSettingsMap& settingsMap)
{
  return new HdGatlingRenderDelegate(settingsMap);
}

void HdGatlingRendererPlugin::DeleteRenderDelegate(HdRenderDelegate* renderDelegate)
{
  delete renderDelegate;
}

bool HdGatlingRendererPlugin::IsSupported() const
{
  return m_isSupported;
}

PXR_NAMESPACE_CLOSE_SCOPE
