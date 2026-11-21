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

#include "MtlxDocumentPatcher.h"

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

  mx::TypedElementPtr _FindSurfaceShaderElement(mx::DocumentPtr doc, const char* name)
  {
    // Find material in document
    auto materialNodes = doc->getMaterialNodes();

    if (materialNodes.size() != 1)
    {
      // HdMtlx generates a new document for each material; there should be a single material
      GB_ERROR("unexpected material count in mtlx document for material {}", name);
      return nullptr;
    }

    mx::NodePtr materialNode = materialNodes[0];

    // Extract surface shader node
    auto surfaceShaderNodes = mx::getShaderNodes(materialNode, mx::SURFACE_SHADER_TYPE_STRING);

    mx::TypedElementPtr renderableElement = materialNode;
    if (!surfaceShaderNodes.empty())
    {
      renderableElement = *surfaceShaderNodes.begin();

      if (surfaceShaderNodes.size() > 1)
      {
        GB_WARN("found multiple surface shaders for material {}; choosing first", name);
      }
    }

    mx::ElementPtr surfaceElement = doc->getDescendant(renderableElement->getNamePath());
    if (!surfaceElement)
    {
      GB_WARN("could not find surface shader for material {}", name);
      return nullptr;
    }

    GB_DEBUG("found surface shader: {}", surfaceElement->getName());
    return surfaceElement->asA<mx::TypedElement>();
  }
}

namespace gtl
{
  McMtlxMdlCodeGen::McMtlxMdlCodeGen(const mx::DocumentPtr mtlxStdLib, const std::string& customNodesPath)
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

    m_docPatcher = std::make_shared<McMtlxDocumentPatcher>(mtlxStdLib, customNodesPath);
  }

  bool McMtlxMdlCodeGen::translate(std::string_view mtlxStr, const char* name, std::string& mdlSrc, std::string& subIdentifier, McOpacityClassification& opacityClass, bool& isAnimated, bool usdPreviewSurfacePresenceOverride)
  {
    try
    {
      mx::DocumentPtr doc = mx::createDocument();
      doc->importLibrary(m_baseDoc);
      mx::readFromXmlString(doc, mtlxStr.data());

      return translate(doc, name, mdlSrc, subIdentifier, opacityClass, isAnimated, usdPreviewSurfacePresenceOverride);
    }
    catch (const std::exception& ex)
    {
      GB_ERROR("exception creating MaterialX document: {}", ex.what());
      return false;
    }
  }

  bool McMtlxMdlCodeGen::translate(const MaterialX::DocumentPtr mtlxDoc, const char* name, std::string& mdlSrc, std::string& subIdentifier, McOpacityClassification& opacityClass, bool& isAnimated, bool usdPreviewSurfacePresenceOverride)
  {
    // Don't cache the context because it is thread-local.
    mx::GenContext context(m_shaderGen);
    context.registerSourceCodeSearchPath(m_mtlxSearchPath);

    mx::GenOptions& contextOptions = context.getOptions();
    contextOptions.targetDistanceUnit = "meter";
    contextOptions.targetColorSpaceOverride = "lin_rec709";

    mx::ShaderPtr shader = nullptr;
    try
    {
      mx::DocumentPtr patchedDoc = m_docPatcher->patch(mtlxDoc, usdPreviewSurfacePresenceOverride);

      if (getenv("GTL_DUMP_MTLX"))
      {
        std::string mtlxSrc = mx::writeToXmlString(patchedDoc);
        GB_LOG("MaterialX source: \n{}", mtlxSrc);
      }

      mx::TypedElementPtr element = _FindSurfaceShaderElement(patchedDoc, name);
      if (!element)
      {
        GB_ERROR("generation failed: surface shader not found");
        return false;
      }

      subIdentifier = element->getName();
      opacityClass = _ClassifySurfaceShaderOpacity(element);
      isAnimated = _HasDocumentTimeNode(patchedDoc);
      shader = m_shaderGen->generate(subIdentifier, element, context);
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
