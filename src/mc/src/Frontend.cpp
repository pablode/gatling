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

#include "Frontend.h"

#include "Material.h"
#include "MdlMaterial.h"
#include "MdlMaterialCompiler.h"
#include "MdlRuntime.h"
#include "MtlxMdlCodeGen.h"
#include "Runtime.h"

#include <filesystem>
#include <assert.h>

namespace
{
  bool _IsCompiledMaterialEmissive(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    mi::base::Handle<const mi::neuraylib::IExpression> expr(compiledMaterial->lookup_sub_expression("surface.emission.intensity"));

    if (expr->get_kind() != mi::neuraylib::IExpression::Kind::EK_CONSTANT)
    {
      return true;
    }

    mi::base::Handle<const mi::neuraylib::IExpression_constant> constExpr(expr.get_interface<const mi::neuraylib::IExpression_constant>());
    mi::base::Handle<const mi::neuraylib::IValue> value(constExpr->get_value());

    if (value->get_kind() != mi::neuraylib::IValue::Kind::VK_COLOR)
    {
      assert(false);
      return true;
    }

    mi::base::Handle<const mi::neuraylib::IValue_color> color(value.get_interface<const mi::neuraylib::IValue_color>());

    if (color->get_size() != 3)
    {
      assert(false);
      return true;
    }

    mi::base::Handle<const mi::neuraylib::IValue_float> v0(color->get_value(0));
    mi::base::Handle<const mi::neuraylib::IValue_float> v1(color->get_value(1));
    mi::base::Handle<const mi::neuraylib::IValue_float> v2(color->get_value(2));

    const float eps = 1e-7f;
    return v0->get_value() > eps || v1->get_value() > eps || v2->get_value() > eps;
  }

  bool _IsCompiledMaterialOpaque(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    return compiledMaterial->get_opacity() == mi::neuraylib::OPACITY_OPAQUE;
  }
}

namespace gtl
{
  McFrontend::McFrontend(const std::vector<std::string>& mdlSearchPaths,
                         const std::vector<std::string>& mtlxSearchPaths,
                         McRuntime& runtime)
  {
    McMdlRuntime& mdlRuntime = runtime.getMdlRuntime();
    m_mdlMaterialCompiler = std::make_shared<McMdlMaterialCompiler>(mdlRuntime, mdlSearchPaths);
    m_mtlxMdlCodeGen = std::make_shared<McMtlxMdlCodeGen>(mtlxSearchPaths);
  }

  McMaterial* McFrontend::createFromMdlStr(std::string_view mdlSrc, std::string_view subIdentifier, bool isOpaque)
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromString(mdlSrc, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    auto mdlMaterial = std::make_shared<McMdlMaterial>();
    mdlMaterial->compiledMaterial = compiledMaterial;

    return new McMaterial{
      .isEmissive = _IsCompiledMaterialEmissive(compiledMaterial),
      .isOpaque = isOpaque,
      .mdlMaterial = mdlMaterial,
      .resourcePathPrefix = "" // no source file
    };
  }

  McMaterial* McFrontend::createFromMtlxStr(std::string_view docStr)
  {
    std::string mdlSrc;
    std::string subIdentifier;
    bool isOpaque;
    if (!m_mtlxMdlCodeGen->translate(docStr, mdlSrc, subIdentifier, isOpaque))
    {
      return nullptr;
    }

    return createFromMdlStr(mdlSrc, subIdentifier, isOpaque);
  }

  McMaterial* McFrontend::createFromMtlxDoc(const MaterialX::DocumentPtr doc)
  {
    std::string mdlSrc;
    std::string subIdentifier;
    bool isOpaque;
    if (!m_mtlxMdlCodeGen->translate(doc, mdlSrc, subIdentifier, isOpaque))
    {
      return nullptr;
    }

    return createFromMdlStr(mdlSrc, subIdentifier, isOpaque);
  }

  McMaterial* McFrontend::createFromMdlFile(std::string_view filePath, std::string_view subIdentifier)
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromFile(filePath, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    namespace fs = std::filesystem;
    std::string resourcePathPrefix = fs::path(filePath).parent_path().string();

    auto mdlMaterial = std::make_shared<McMdlMaterial>();
    mdlMaterial->compiledMaterial = compiledMaterial;

    return new McMaterial {
      .isEmissive = _IsCompiledMaterialEmissive(compiledMaterial),
      .isOpaque = _IsCompiledMaterialOpaque(compiledMaterial),
      .mdlMaterial = mdlMaterial,
      .resourcePathPrefix = resourcePathPrefix
    };
  }
}
