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

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXCore/Material.h>
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
  using namespace gtl;

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

    // The MaterialX MDL generator does not import the scene module, which the custom parameter
    // nodes use for scene data lookups.
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
