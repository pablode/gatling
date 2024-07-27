//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <gtl/gb/Fmt.h>
#include <gtl/gt/LogFlushListener.h>

#include <pxr/base/tf/getenv.h>
#include <pxr/base/tf/setenv.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/rendererPluginHandle.h>
#include <pxr/imaging/hd/rendererPlugin.h>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace gtl;

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  (HdGatlingRendererPlugin)
);

int main(int argc, char** argv)
{
  // Prevent global gatling installation from messing with test results
  TfSetenv("PXR_DISABLE_STANDARD_PLUG_SEARCH_PATH", "1");

  // Append PATH to find hdGatling.dll on Windows
#if defined(ARCH_OS_WINDOWS)
  {
    const char* PATH_NAME = "PATH";
    std::string newPath = GB_FMT("{};{}", TfGetenv(PATH_NAME), HDGATLING_INSTALL_DIR);
    TfSetenv(PATH_NAME, newPath.c_str());
  }
#endif

  // Register plugin
  std::string plugInfoDir = GB_FMT("{}/hdGatling/resources", HDGATLING_INSTALL_DIR);

  PlugRegistry& plugRegistry = PlugRegistry::GetInstance();
  plugRegistry.RegisterPlugins(plugInfoDir);

  // Run tests
  doctest::Context context;
  context.applyCommandLine(argc, argv);

  int result = context.run();

  return result;
}

class GraphicalTestFixture
{
public:
  GraphicalTestFixture()
  {
    HdRendererPluginRegistry& pluginRegistry = HdRendererPluginRegistry::GetInstance();
    m_plugin = pluginRegistry.GetOrCreateRendererPlugin(_tokens->HdGatlingRendererPlugin);
    REQUIRE(m_plugin);
    REQUIRE(m_plugin->IsSupported());

    m_renderDelegate = m_plugin->CreateRenderDelegate();
    REQUIRE(m_renderDelegate);

    // Register listener after delegate initializes logger
    doctest::registerReporter<GtLogFlushListener>("LogFlush", 1, false);
  }

  ~GraphicalTestFixture()
  {
    m_plugin->DeleteRenderDelegate(m_renderDelegate);
  }

private:
  HdRendererPluginHandle m_plugin;
  HdRenderDelegate* m_renderDelegate = nullptr;
};

TEST_CASE_FIXTURE(GraphicalTestFixture, "CreateDelegate")
{
}
