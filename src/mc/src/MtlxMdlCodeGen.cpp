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
#include <Log.h>

namespace mx = MaterialX;

namespace
{
  bool _IsBxdfWithInputValue(mx::NodePtr node,
                             const std::string& category,
                             const std::string& inputName,
                             mx::ValuePtr expectedValue)
  {
    if (!node || node->getCategory() != category)
    {
      return false;
    }

    mx::ValuePtr inputValue = node->getInputValue(inputName);
    float floatEps = 0.0001f;

    if (inputValue && inputValue->isA<float>() && expectedValue->isA<float>())
    {
      return fabs(inputValue->asA<float>() - expectedValue->asA<float>()) < floatEps;
    }

    if (inputValue && inputValue->isA<mx::Color3>() && expectedValue->isA<mx::Color3>())
    {
      mx::Color3 diff = inputValue->asA<mx::Color3>() - expectedValue->asA<mx::Color3>();
      return fabs(diff[0]) < floatEps && fabs(diff[1]) < floatEps && fabs(diff[2]) < floatEps;
    }

    if (inputValue && inputValue->isA<int>() && expectedValue->isA<int>())
    {
      return inputValue->asA<int>() == expectedValue->asA<int>();
    }

    return false;
  }

  bool _IsSurfaceShaderOpaque(mx::TypedElementPtr element)
  {
    mx::NodePtr node = element->asA<mx::Node>();

    if (_IsBxdfWithInputValue(node, "UsdPreviewSurface", "opacity", mx::Value::createValue(1.0f)))
    {
      return true;
    }

    if (_IsBxdfWithInputValue(node, "standard_surface", "opacity", mx::Value::createValue(mx::Color3(1.0f))) &&
        _IsBxdfWithInputValue(node, "standard_surface", "transmission", mx::Value::createValue(0.0f)) &&
        _IsBxdfWithInputValue(node, "standard_surface", "subsurface", mx::Value::createValue(0.0f)))
    {
      return true;
    }

    if (_IsBxdfWithInputValue(node, "gltf_pbr", "alpha_mode", mx::Value::createValue(0)) &&
        _IsBxdfWithInputValue(node, "gltf_pbr", "transmission", mx::Value::createValue(0.0f)))
    {
      return true;
    }

    if (_IsBxdfWithInputValue(node, "open_pbr_surface", "geometry_opacity", mx::Value::createValue(mx::Color3(1.0f))) &&
        _IsBxdfWithInputValue(node, "open_pbr_surface", "transmission_weight", mx::Value::createValue(0.0f)) &&
        _IsBxdfWithInputValue(node, "open_pbr_surface", "subsurface_weight", mx::Value::createValue(0.0f)))
    {
      return true;
    }

    // Use MaterialX helper function as fallback (not accurate, has false positives)
    return !mx::isTransparentSurface(element);
  }

  mx::TypedElementPtr _FindSurfaceShaderElement(mx::DocumentPtr doc)
  {
    // Find renderable element.
    std::vector<mx::TypedElementPtr> renderableElements;
#if (MATERIALX_MAJOR_VERSION > 1) || \
    (MATERIALX_MAJOR_VERSION == 1 && MATERIALX_MINOR_VERSION > 38) || \
    (MATERIALX_MAJOR_VERSION == 1 && MATERIALX_MINOR_VERSION == 38 && MATERIALX_BUILD_VERSION > 8)
    renderableElements = mx::findRenderableElements(doc);
#else
    mx::findRenderableElements(doc, renderableElements);
#endif

    for (mx::TypedElementPtr elem : renderableElements)
    {
      // Extract surface shader node.
      mx::TypedElementPtr renderableElement = renderableElements.at(0);
      mx::NodePtr node = renderableElement->asA<mx::Node>();

      if (node && node->getType() == mx::MATERIAL_TYPE_STRING)
      {
        auto shaderNodes = mx::getShaderNodes(node, mx::SURFACE_SHADER_TYPE_STRING);
        if (!shaderNodes.empty())
        {
          renderableElement = *shaderNodes.begin();
        }
      }

      mx::ElementPtr surfaceElement = doc->getDescendant(renderableElement->getNamePath());
      if (!surfaceElement)
      {
        return nullptr;
      }

      return surfaceElement->asA<mx::TypedElement>();
    }

    return nullptr;
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

  bool McMtlxMdlCodeGen::translate(std::string_view mtlxStr, std::string& mdlSrc, std::string& subIdentifier, bool& isOpaque)
  {
    try
    {
      mx::DocumentPtr doc = mx::createDocument();
      doc->importLibrary(m_baseDoc);
      mx::readFromXmlString(doc, mtlxStr.data());

      return translate(doc, mdlSrc, subIdentifier, isOpaque);
    }
    catch (const std::exception& ex)
    {
      GB_ERROR("exception creating MaterialX document: {}", ex.what());
      return false;
    }
  }

  bool McMtlxMdlCodeGen::translate(MaterialX::DocumentPtr mtlxDoc, std::string& mdlSrc, std::string& subIdentifier, bool& isOpaque)
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
      McMtlxDocumentPatcher patcher;
      patcher.patch(mtlxDoc);

      if (getenv("GATLING_DUMP_MTLX"))
      {
        std::string mtlxSrc = mx::writeToXmlString(mtlxDoc);
        GB_LOG("MaterialX source: \n{}", mtlxSrc);
      }

      mx::TypedElementPtr element = _FindSurfaceShaderElement(mtlxDoc);
      if (!element)
      {
        GB_ERROR("generation failed: surface shader not found");
        return false;
      }

      subIdentifier = element->getName();
      isOpaque = _IsSurfaceShaderOpaque(element);
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

    if (getenv("GATLING_DUMP_MDL"))
    {
      GB_LOG("MDL source: \n{}", mdlSrc);
    }

    return true;
  }
}
