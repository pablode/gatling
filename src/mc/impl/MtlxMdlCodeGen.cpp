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

  bool _HasSurfaceShaderNoCutoutTransparency(mx::TypedElementPtr element)
  {
    mx::NodePtr node = element->asA<mx::Node>();

    if (_IsBxdfWithInputValue(node, "UsdPreviewSurface", "opacity", mx::Value::createValue(1.0f)))
    {
      return true;
    }

    if (_IsBxdfWithInputValue(node, "standard_surface", "opacity", mx::Value::createValue(mx::Color3(1.0f))))
    {
      return true;
    }

    if (_IsBxdfWithInputValue(node, "gltf_pbr", "alpha_mode", mx::Value::createValue(0 /* OPAQUE */)) ||
        _IsBxdfWithInputValue(node, "gltf_pbr", "alpha_mode", mx::Value::createValue(2 /* BLEND */)))
    {
      return true;
    }

    if (_IsBxdfWithInputValue(node, "open_pbr_surface", "geometry_opacity", mx::Value::createValue(1.0f)))
    {
      return true;
    }

    // Use MaterialX helper function as fallback (not accurate, has false positives)
    return !mx::isTransparentSurface(element);
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
                                   bool& hasCutoutTransparency)
  {
    // Don't cache the context because it is thread-local.
    mx::GenContext context(m_shaderGen);

    mx::GenOptions& contextOptions = context.getOptions();
    contextOptions.targetDistanceUnit = "meter";
    contextOptions.targetColorSpaceOverride = "lin_rec709";

    mx::ShaderPtr shader = nullptr;
    try
    {
      if (getenv("GATLING_DUMP_MTLX"))
      {
        std::string mtlxSrc = mx::writeToXmlString(mtlxDoc);
        GB_LOG("MaterialX source: \n{}", mtlxSrc);
      }

      subIdentifier = surfaceShader->getName();
      hasCutoutTransparency = !_HasSurfaceShaderNoCutoutTransparency(surfaceShader);
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

    if (getenv("GATLING_DUMP_MDL"))
    {
      GB_LOG("MDL source: \n{}", mdlSrc);
    }

    return true;
  }
}
