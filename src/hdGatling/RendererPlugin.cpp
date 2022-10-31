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

#include "RendererPlugin.h"
#include "RenderDelegate.h"

#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/thisPlugin.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/packageUtils.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

class UsdzAssetReader : public GiAssetReader
{
private:
  struct UsdzAsset
  {
    size_t size;
    std::shared_ptr<const char> buffer;
  };

public:
  GiAsset* open(const char* path) override
  {
    auto resolvedPath = ArResolvedPath(path);
    if (!ArIsPackageRelativePath(resolvedPath)) {
      // Only read USDZ files with this reader, fall back
      // to memory-mapping default for everything else.
      return nullptr;
    }

    ArResolver& resolver = ArGetResolver();
    auto asset = resolver.OpenAsset(resolvedPath);
    if (!asset) {
      return nullptr;
    }

    auto iasset = new UsdzAsset;
    iasset->size = asset->GetSize();
    iasset->buffer = asset->GetBuffer();
    return (GiAsset*) iasset;
  }

  size_t size(const GiAsset* asset) const override
  {
    auto iasset = (UsdzAsset*) asset;
    return iasset->size;
  }

  void* data(const GiAsset* asset) const override
  {
    auto iasset = (UsdzAsset*) asset;
    return (void*) iasset->buffer.get();
  }

  void close(GiAsset* asset) override
  {
    auto iasset = (UsdzAsset*) asset;
    delete iasset;
  }
};

TF_REGISTRY_FUNCTION(TfType)
{
  HdRendererPluginRegistry::Define<HdGatlingRendererPlugin>();
}

HdGatlingRendererPlugin::HdGatlingRendererPlugin()
{
  PlugPluginPtr plugin = PLUG_THIS_PLUGIN;

  const std::string& resourcePath = plugin->GetResourcePath();
  std::string shaderPath = resourcePath + "/shaders";
  std::string mdlLibPath = resourcePath + "/mdl";
  std::string mtlxLibPath = resourcePath + "/materialx";

  m_translator = std::make_unique<MaterialNetworkTranslator>(mtlxLibPath);

  gi_init_params params;
  params.resource_path = resourcePath.c_str();
  params.shader_path = shaderPath.c_str();
  params.mtlx_lib_path = mtlxLibPath.c_str();
  params.mdl_lib_path = mdlLibPath.c_str();

  m_isSupported = (giInitialize(&params) == GI_OK);
  if (!m_isSupported)
  {
    return;
  }

  m_usdzAssetReader = std::make_unique<UsdzAssetReader>();
  giRegisterAssetReader(m_usdzAssetReader.get());
}

HdGatlingRendererPlugin::~HdGatlingRendererPlugin()
{
  if (!m_isSupported)
  {
    return;
  }

  m_usdzAssetReader.reset();
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
