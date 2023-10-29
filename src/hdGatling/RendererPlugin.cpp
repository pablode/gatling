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
#include "MaterialNetworkCompiler.h"

#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/thisPlugin.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/packageUtils.h>
#include <pxr/usd/usdMtlx/utils.h>

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

bool _TryInitGi(const std::vector<std::string>& mtlxSearchPaths)
{
  PlugPluginPtr plugin = PLUG_THIS_PLUGIN;

  const std::string& resourcePath = plugin->GetResourcePath();
  std::string shaderPath = resourcePath + "/shaders";

  // USD installs the 'source/MaterialXGenMdl/mdl' folder to the MaterialX 'libraries' dir
  std::vector<std::string> mdlSearchPaths = UsdMtlxStandardLibraryPaths();
  // In addition, we also install Omni* MDL files to support TurboSquid files
  mdlSearchPaths.push_back(resourcePath);

  // The 'mdl' folder is not part of the MDL package paths
  for (std::string& s : mdlSearchPaths)
  {
    s += "/mdl";
  }

  GiInitParams params = {
    .resourcePath = resourcePath.c_str(),
    .shaderPath = shaderPath.c_str(),
    .mdlSearchPaths = mdlSearchPaths,
    .mtlxSearchPaths = mtlxSearchPaths
  };

  return giInitialize(&params) == GI_OK;
}

HdGatlingRendererPlugin::HdGatlingRendererPlugin()
{
  const std::vector<std::string>& mtlxSearchPaths = UsdMtlxSearchPaths();

  m_isSupported = _TryInitGi(mtlxSearchPaths);

  if (!m_isSupported)
  {
    return;
  }

  m_translator = std::make_unique<MaterialNetworkCompiler>(mtlxSearchPaths);

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

  return CreateRenderDelegate(settingsMap);
}

HdRenderDelegate* HdGatlingRendererPlugin::CreateRenderDelegate(const HdRenderSettingsMap& settingsMap)
{
  PlugPluginPtr plugin = PLUG_THIS_PLUGIN;

  const std::string& resourcePath = plugin->GetResourcePath();

  return new HdGatlingRenderDelegate(settingsMap, *m_translator, resourcePath);
}

void HdGatlingRendererPlugin::DeleteRenderDelegate(HdRenderDelegate* renderDelegate)
{
  delete renderDelegate;
}

#if PXR_VERSION >= 2302
bool HdGatlingRendererPlugin::IsSupported(bool gpuEnabled) const
#else
bool HdGatlingRendererPlugin::IsSupported() const
#endif
{
  return m_isSupported;
}

PXR_NAMESPACE_CLOSE_SCOPE
