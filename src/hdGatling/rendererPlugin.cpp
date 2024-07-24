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

#include "rendererPlugin.h"
#include "renderDelegate.h"
#include "materialNetworkCompiler.h"

#include <pxr/imaging/hdMtlx/hdMtlx.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/thisPlugin.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/usdMtlx/utils.h>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <gtl/gi/Gi.h>

using namespace gtl;
namespace mx = MaterialX;

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
  bool _TryInitGi(const mx::DocumentPtr mtlxStdLib)
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
      .shaderPath = shaderPath.c_str(),
      .mdlRuntimePath = resourcePath.c_str(),
      .mdlSearchPaths = mdlSearchPaths,
      .mtlxStdLib = mtlxStdLib
    };
    return giInitialize(params) == GiStatus::Ok;
  }

  mx::DocumentPtr _LoadMtlxStdLib()
  {
    mx::DocumentPtr mtlxStdLib = mx::createDocument();

    mx::FileSearchPath fileSearchPaths;
    for (const std::string& s : UsdMtlxSearchPaths())
    {
      fileSearchPaths.append(mx::FilePath(s));
    }

    mx::FilePathVec libFolders; // All directories if left empty.
    mx::loadLibraries(libFolders, fileSearchPaths, mtlxStdLib);

    return mtlxStdLib;
  }
}

// Needs to be defined in pxr namespace due to being publicly declared in header
class ArAssetReader : public GiAssetReader
{
private:
  struct _ArAsset
  {
    size_t size;
    std::shared_ptr<const char> buffer;
  };

public:
  GiAsset* open(const char* path) override
  {
    ArResolver& resolver = ArGetResolver();

    ArResolvedPath resolvedPath = resolver.Resolve(path);
    if (!resolvedPath)
    {
      return nullptr;
    }

    auto asset = resolver.OpenAsset(resolvedPath);
    if (!asset) {
      return nullptr;
    }

    auto iasset = new _ArAsset;
    iasset->size = asset->GetSize();
    iasset->buffer = asset->GetBuffer();
    return (GiAsset*) iasset;
  }

  size_t size(const GiAsset* asset) const override
  {
    auto iasset = (_ArAsset*) asset;
    return iasset->size;
  }

  void* data(const GiAsset* asset) const override
  {
    auto iasset = (_ArAsset*) asset;
    return (void*) iasset->buffer.get();
  }

  void close(GiAsset* asset) override
  {
    auto iasset = (_ArAsset*) asset;
    delete iasset;
  }
};

TF_REGISTRY_FUNCTION(TfType)
{
  HdRendererPluginRegistry::Define<HdGatlingRendererPlugin>();
}

HdGatlingRendererPlugin::HdGatlingRendererPlugin()
{
#if PXR_VERSION > 2311
  mx::DocumentPtr mtlxStdLib = HdMtlxStdLibraries();
#else
  mx::DocumentPtr mtlxStdLib = _LoadMtlxStdLib();
#endif

  _isSupported = _TryInitGi(mtlxStdLib);

  if (!_isSupported)
  {
    return;
  }

  _materialNetworkCompiler = std::make_unique<MaterialNetworkCompiler>(mtlxStdLib);

  _arAssetReader = std::make_unique<ArAssetReader>();
  giRegisterAssetReader(_arAssetReader.get());
}

HdGatlingRendererPlugin::~HdGatlingRendererPlugin()
{
  if (!_isSupported)
  {
    return;
  }

  _arAssetReader.reset();
  giTerminate();
}

HdRenderDelegate* HdGatlingRendererPlugin::CreateRenderDelegate()
{
  HdRenderSettingsMap settingsMap;

  return CreateRenderDelegate(settingsMap);
}

HdRenderDelegate* HdGatlingRendererPlugin::CreateRenderDelegate(const HdRenderSettingsMap& settingsMap)
{
  if (!_isSupported)
  {
    return nullptr;
  }

  PlugPluginPtr plugin = PLUG_THIS_PLUGIN;

  const std::string& resourcePath = plugin->GetResourcePath();

  return new HdGatlingRenderDelegate(settingsMap, *_materialNetworkCompiler, resourcePath);
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
  return _isSupported;
}

PXR_NAMESPACE_CLOSE_SCOPE
