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

#include "MtlxMdlCodeGen.h"

#include "MtlxDocPatch.h"
#include "MtlxDocOps.h"

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXCore/Material.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/GenOptions.h>
#include <MaterialXGenShader/Library.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenMdl/MdlShaderGenerator.h>
#include <gtl/gb/Log.h>

#ifdef MC_TEST_EXECUTABLE
#include <doctest/doctest.h>
#include <functional>
#include <string_view>
#endif

namespace mx = MaterialX;

namespace
{
  using namespace gtl;

  bool _IsNodeInputModified(mx::NodePtr node, const char* inputName, mx::ValuePtr defaultValue)
  {
    if (node->getConnectedNode(inputName) != nullptr)
    {
      return true; // custom logic hooked up
    }

    mx::ValuePtr inputValue = node->getInputValue(inputName);
    if (!inputValue)
    {
      return false;
    }

    auto isApproxEqual = [](float a, float b) { return fabs(a - b) < 0.0001f; };

    if (inputValue->isA<float>() && defaultValue->isA<float>())
    {
      return !isApproxEqual(inputValue->asA<float>(), defaultValue->asA<float>());
    }

    if (inputValue->isA<mx::Color3>() && defaultValue->isA<mx::Color3>())
    {
      mx::Color3 value = inputValue->asA<mx::Color3>();
      mx::Color3 defaultColor = defaultValue->asA<mx::Color3>();

      return !isApproxEqual(value[0], defaultColor[0]) ||
             !isApproxEqual(value[1], defaultColor[1]) ||
             !isApproxEqual(value[2], defaultColor[2]);
    }

    if (inputValue->isA<int>() && defaultValue->isA<int>())
    {
      return inputValue->asA<int>() != defaultValue->asA<int>();
    }

    return false;
  }

  McOpacityClassification _ClassifySurfaceShaderOpacity(mx::TypedElementPtr element)
  {
    mx::NodePtr node = element->asA<mx::Node>();
    const std::string& category = node->getCategory();

    // Accurate detection for known uber-BSDFs
    if (category == "gltf_pbr")
    {
      mx::ValuePtr inputValue = node->getInputValue("alpha_mode");
      int alphaMode = (inputValue && inputValue->isA<int>()) ? inputValue->asA<int>() : 0;

      if (alphaMode == 1 /*MASK*/)
      {
        return McOpacityClassification::Cutout;
      }
      if (alphaMode == 2 /*BLEND*/ ||
          _IsNodeInputModified(node, "transmission", mx::Value::createValue(0.0f)))
      {
        return McOpacityClassification::Transparent;
      }

      return McOpacityClassification::Opaque;
    }
    else if (category == "UsdPreviewSurface")
    {
      bool opacityModified = _IsNodeInputModified(node, "opacity", mx::Value::createValue(1.0f));
      bool presenceMode = _IsNodeInputModified(node, "opacityMode", mx::Value::createValue(0 /* TRANSPARENT */));

      // A threshold or presence mode turns opacity into a cutout; otherwise it transmits.
      if (_IsNodeInputModified(node, "opacityThreshold", mx::Value::createValue(0.0f)) ||
          (presenceMode && opacityModified))
      {
        return McOpacityClassification::Cutout;
      }
      if (opacityModified)
      {
        return McOpacityClassification::Transparent;
      }

      return McOpacityClassification::Opaque;
    }
    else if (category == "standard_surface")
    {
      if (_IsNodeInputModified(node, "transmission", mx::Value::createValue(0.0f)) ||
          _IsNodeInputModified(node, "subsurface", mx::Value::createValue(0.0f)))
      {
        return McOpacityClassification::Transparent;
      }
      if (_IsNodeInputModified(node, "opacity", mx::Value::createValue(mx::Color3(1.0f))))
      {
        return McOpacityClassification::Cutout;
      }
      return McOpacityClassification::Opaque;
    }
    else if (category == "open_pbr_surface")
    {
      if (_IsNodeInputModified(node, "transmission_weight", mx::Value::createValue(0.0f)) ||
          _IsNodeInputModified(node, "subsurface_weight", mx::Value::createValue(0.0f)))
      {
        return McOpacityClassification::Transparent;
      }
      if (_IsNodeInputModified(node, "geometry_opacity", mx::Value::createValue(1.0f)))
      {
        return McOpacityClassification::Cutout;
      }
      return McOpacityClassification::Opaque;
    }

    // Use MaterialX helper function as fallback and assume worst case (not accurate, has false positives)
    return mx::isTransparentSurface(element) ? McOpacityClassification::Transparent : McOpacityClassification::Opaque;
  }

  bool _HasDocumentTimeNode(mx::DocumentPtr doc)
  {
    for (mx::NodeGraphPtr graph : doc->getNodeGraphs())
    {
      if (graph->getActiveSourceUri() != doc->getSourceUri())
      {
        continue;
      }

      for (auto treeIt = graph->traverseTree(); treeIt != mx::TreeIterator::end(); ++treeIt)
      {
        mx::ElementPtr elem = treeIt.getElement();

        mx::NodePtr node = elem->asA<mx::Node>();
        if (!node)
        {
          continue;
        }

        const mx::string& category = node->getCategory();
        if (category == "time")
        {
          return true;
        }
      }
    }

    return false;
  }
}

namespace gtl
{
  McMtlxMdlCodeGen::McMtlxMdlCodeGen(const mx::DocumentPtr mtlxStdLib)
  {
    // Init shadergen.
    m_shaderGen = mx::MdlShaderGenerator::create();
    std::string target = m_shaderGen->getTarget();

    // Import stdlib.
    m_baseDoc = mx::createDocument();
    m_baseDoc->importLibrary(mtlxStdLib);

    // Color management.
    mx::DefaultColorManagementSystemPtr colorSystem = mx::DefaultColorManagementSystem::create(target);
    colorSystem->loadLibrary(m_baseDoc);
    m_shaderGen->setColorManagementSystem(colorSystem);

    // Unit management.
    mx::UnitSystemPtr unitSystem = mx::UnitSystem::create(target);
    unitSystem->loadLibrary(m_baseDoc);

    mx::UnitConverterRegistryPtr unitRegistry = mx::UnitConverterRegistry::create();
    mx::UnitTypeDefPtr distanceTypeDef = m_baseDoc->getUnitTypeDef("distance");
    unitRegistry->addUnitConverter(distanceTypeDef, mx::LinearUnitConverter::create(distanceTypeDef));
    mx::UnitTypeDefPtr angleTypeDef = m_baseDoc->getUnitTypeDef("angle");
    unitRegistry->addUnitConverter(angleTypeDef, mx::LinearUnitConverter::create(angleTypeDef));

    unitSystem->setUnitConverterRegistry(unitRegistry);
    m_shaderGen->setUnitSystem(unitSystem);
  }

  bool McMtlxMdlCodeGen::translate(const mx::DocumentPtr& mtlxDoc,
                                   const mx::NodePtr& surfaceShader,
                                   std::string& mdlSrc,
                                   std::string& subIdentifier,
                                   McOpacityClassification& opacityClass,
                                   bool& isAnimated)
  {
    // Don't cache the context because it is thread-local.
    mx::GenContext context(m_shaderGen);

    mx::GenOptions& contextOptions = context.getOptions();
    contextOptions.targetDistanceUnit = "meter";
    contextOptions.targetColorSpaceOverride = "lin_rec709";

    mx::ShaderPtr shader = nullptr;
    try
    {
      if (getenv("GTL_DUMP_MTLX"))
      {
        std::string mtlxSrc = mx::writeToXmlString(mtlxDoc);
        GB_LOG("MaterialX source: \n{}", mtlxSrc);
      }

      subIdentifier = surfaceShader->getName();
      opacityClass = _ClassifySurfaceShaderOpacity(surfaceShader);
      isAnimated = _HasDocumentTimeNode(mtlxDoc);
      shader = m_shaderGen->generate(subIdentifier, surfaceShader, context);
    }
    catch (const std::exception& ex)
    {
      GB_ERROR("exception generating MDL code: {}", ex.what());
    }

    if (!shader)
    {
      return false;
    }

    mx::ShaderStage pixelStage = shader->getStage(mx::Stage::PIXEL);
    mdlSrc = pixelStage.getSourceCode();

    // The MDL generator does not import the scene module used by our custom NodeDef.
    size_t versionLineEnd = mdlSrc.find('\n');
    size_t insertPos = versionLineEnd == std::string::npos ? 0 : versionLineEnd + 1;
    mdlSrc.insert(insertPos, "\nimport ::scene::*;\n");

    if (getenv("GTL_DUMP_MDL"))
    {
      GB_LOG("MDL source: \n{}", mdlSrc);
    }

    return true;
  }
}

#ifdef MC_TEST_EXECUTABLE
namespace
{
  TEST_CASE("MtlxMdlCodeGen.ClassifySurfaceShaderOpacity")
  {
    struct ClassificationCase
    {
      std::string_view baseName;
      std::string_view testName;
      std::function<mx::NodePtr(mx::DocumentPtr)> createShader;
      McOpacityClassification expected;
    };

    const static ClassificationCase testCases[] = {
      // UsdPreviewSurface
      { "UsdPreviewSurface", "default", [](mx::DocumentPtr doc) {
          return doc->addNode("UsdPreviewSurface", "S", mx::SURFACE_SHADER_TYPE_STRING);
        }, McOpacityClassification::Opaque },
      { "UsdPreviewSurface", "opaque", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("UsdPreviewSurface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", 1.0f);
          return node;
        }, McOpacityClassification::Opaque },
      { "UsdPreviewSurface", "blend", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("UsdPreviewSurface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", 0.5f);
          return node;
        }, McOpacityClassification::Transparent },
      { "UsdPreviewSurface", "connected opacity", [](mx::DocumentPtr doc) {
          auto constant = doc->addNode("constant", "C", "float");
          constant->setInputValue("value", 0.5f);
          auto node = doc->addNode("UsdPreviewSurface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setConnectedNode("opacity", constant);
          return node;
        }, McOpacityClassification::Transparent },
      { "UsdPreviewSurface", "cutout", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("UsdPreviewSurface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", 0.5f);
          node->setInputValue("opacityThreshold", 0.5f);
          return node;
        }, McOpacityClassification::Cutout },
      { "UsdPreviewSurface", "presence", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("UsdPreviewSurface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", 0.5f);
          node->setInputValue("opacityMode", 1);
          return node;
        }, McOpacityClassification::Cutout },
      { "UsdPreviewSurface", "presence opaque", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("UsdPreviewSurface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacityMode", 1);
          return node;
        }, McOpacityClassification::Opaque },

      // standard_surface
      { "standard_surface", "opaque", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("standard_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", mx::Color3(1.0f));
          return node;
        }, McOpacityClassification::Opaque },
      { "standard_surface", "blend", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("standard_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", mx::Color3(0.5f));
          return node;
        }, McOpacityClassification::Cutout },
      { "standard_surface", "connected opacity", [](mx::DocumentPtr doc) {
          auto constant = doc->addNode("constant", "C", "color3");
          constant->setInputValue("value", mx::Color3(1.0f));
          auto node = doc->addNode("standard_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setConnectedNode("opacity", constant);
          return node;
        }, McOpacityClassification::Cutout },
      { "standard_surface", "transmission", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("standard_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("transmission", 1.0f);
          return node;
        }, McOpacityClassification::Transparent },
      { "standard_surface", "subsurface", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("standard_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("subsurface", 1.0f);
          return node;
        }, McOpacityClassification::Transparent },
      { "standard_surface", "opaque opacity with transmission", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("standard_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", mx::Color3(1.0f));
          node->setInputValue("transmission", 1.0f);
          return node;
        }, McOpacityClassification::Transparent },

      // gltf_pbr
      { "gltf_pbr", "opaque", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("gltf_pbr", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("alpha_mode", 0);
          return node;
        }, McOpacityClassification::Opaque },
      { "gltf_pbr", "mask", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("gltf_pbr", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("alpha_mode", 1);
          return node;
        }, McOpacityClassification::Cutout },
      { "gltf_pbr", "blend", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("gltf_pbr", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("alpha_mode", 2);
          return node;
        }, McOpacityClassification::Transparent },
      { "gltf_pbr", "transmission", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("gltf_pbr", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("transmission", 1.0f);
          return node;
        }, McOpacityClassification::Transparent },

      // open_pbr_surface
      { "open_pbr_surface", "opaque", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("open_pbr_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("geometry_opacity", 1.0f);
          return node;
        }, McOpacityClassification::Opaque },
      { "open_pbr_surface", "blend", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("open_pbr_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("geometry_opacity", 0.5f);
          return node;
        }, McOpacityClassification::Cutout },
      { "open_pbr_surface", "connected opacity", [](mx::DocumentPtr doc) {
          auto constant = doc->addNode("constant", "C", "float");
          constant->setInputValue("value", 0.5f);
          auto node = doc->addNode("open_pbr_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setConnectedNode("geometry_opacity", constant);
          return node;
        }, McOpacityClassification::Cutout },
      { "open_pbr_surface", "transmission", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("open_pbr_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("transmission_weight", 1.0f);
          return node;
        }, McOpacityClassification::Transparent },
      { "open_pbr_surface", "subsurface", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("open_pbr_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("subsurface_weight", 1.0f);
          return node;
        }, McOpacityClassification::Transparent },
      { "open_pbr_surface", "opaque opacity with transmission", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("open_pbr_surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("geometry_opacity", 1.0f);
          node->setInputValue("transmission_weight", 1.0f);
          return node;
        }, McOpacityClassification::Transparent },

      // Custom graphs
      { "custom", "simple opaque surface", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", 1.0f);
          return node;
        }, McOpacityClassification::Opaque },
      { "custom", "simple transparent surface", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("opacity", 0.5f);
          return node;
        }, McOpacityClassification::Transparent },
      { "custom", "simple transmissive surface", [](mx::DocumentPtr doc) {
          auto node = doc->addNode("surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setInputValue("transmission", 1.0f);
          return node;
        }, McOpacityClassification::Transparent },
#if MATERIALX_VERSION >= 13930 // MaterialX fallback behaviour changed
      { "custom", "connected opacity", [](mx::DocumentPtr doc) {
          auto constant = doc->addNode("constant", "C", "float");
          constant->setInputValue("value", 0.5f);
          auto node = doc->addNode("surface", "S", mx::SURFACE_SHADER_TYPE_STRING);
          node->setConnectedNode("opacity", constant);
          return node;
        }, McOpacityClassification::Transparent },
#endif
      { "custom", "opaque mix", [](mx::DocumentPtr doc) {
          auto fg = doc->addNode("surface", "Fg", mx::SURFACE_SHADER_TYPE_STRING);
          auto bg = doc->addNode("surface", "Bg", mx::SURFACE_SHADER_TYPE_STRING);
          auto mix = doc->addNode("mix", "Mix", mx::SURFACE_SHADER_TYPE_STRING);
          mix->setConnectedNode("fg", fg);
          mix->setConnectedNode("bg", bg);
          return mix;
        }, McOpacityClassification::Opaque },
      { "custom", "transparent mix", [](mx::DocumentPtr doc) {
          auto fg = doc->addNode("surface", "Fg", mx::SURFACE_SHADER_TYPE_STRING);
          fg->setInputValue("opacity", 0.5f);
          auto bg = doc->addNode("surface", "Bg", mx::SURFACE_SHADER_TYPE_STRING);
          auto mix = doc->addNode("mix", "Mix", mx::SURFACE_SHADER_TYPE_STRING);
          mix->setConnectedNode("fg", fg);
          mix->setConnectedNode("bg", bg);
          return mix;
        }, McOpacityClassification::Transparent },
    };

    for (const ClassificationCase& testCase : testCases)
    {
      mx::DocumentPtr doc = mx::createDocument();
      mx::NodePtr shader = testCase.createShader(doc);
      REQUIRE(shader);

      McOpacityClassification actual = _ClassifySurfaceShaderOpacity(shader);
      CHECK_MESSAGE(actual == testCase.expected, "opacity classification mismatch: ",
        testCase.baseName, " / ", testCase.testName);
    }
  }

  TEST_CASE("MtlxMdlCodeGen.HasDocumentTimeNode")
  {
    {
      mx::DocumentPtr doc = mx::createDocument();
      CHECK_FALSE(_HasDocumentTimeNode(doc));
    }
    {
      mx::DocumentPtr doc = mx::createDocument();
      doc->addNodeGraph("NG")->addNode("constant", "C", "float");
      CHECK_FALSE(_HasDocumentTimeNode(doc));
    }
    {
      mx::DocumentPtr doc = mx::createDocument();
      doc->addNodeGraph("NG")->addNode("time", "T", "float");
      CHECK(_HasDocumentTimeNode(doc));
    }
  }
}
#endif
