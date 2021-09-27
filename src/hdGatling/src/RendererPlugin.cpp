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
  const std::string& resourcePath = plugin->GetResourcePath();
  printf("Resource path %s\n", resourcePath.c_str());

  std::string mtlxlibPath = resourcePath + "/mtlxlib";
  m_translator = std::make_unique<MaterialNetworkTranslator>(mtlxlibPath);

  int initResult = giInitialize(resourcePath.c_str());
  m_isSupported = (initResult == GI_OK);
}

HdGatlingRendererPlugin::~HdGatlingRendererPlugin()
{
  giTerminate();
}

HdRenderDelegate* HdGatlingRendererPlugin::CreateRenderDelegate()
{
  HdRenderSettingsMap settingsMap;

  return new HdGatlingRenderDelegate(settingsMap, *m_translator);
}

HdRenderDelegate* HdGatlingRendererPlugin::CreateRenderDelegate(const HdRenderSettingsMap& settingsMap)
{
  return new HdGatlingRenderDelegate(settingsMap, *m_translator);
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
