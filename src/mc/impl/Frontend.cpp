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
  bool _IsExpressionBlackColor(mi::base::Handle<const mi::neuraylib::IExpression> expr)
  {
    if (expr->get_kind() != mi::neuraylib::IExpression::Kind::EK_CONSTANT)
    {
      return false;
    }

    mi::base::Handle<const mi::neuraylib::IExpression_constant> constExpr(expr.get_interface<const mi::neuraylib::IExpression_constant>());
    mi::base::Handle<const mi::neuraylib::IValue_color> value(constExpr->get_value<mi::neuraylib::IValue_color>());

    if (!value)
    {
      return false;
    }

    for (mi::Size i = 0; i < value->get_size(); i++)
    {
      mi::base::Handle<const mi::neuraylib::IValue_float> c(value->get_value(i));

      const float eps = 1e-7f;

      if (c->get_value() > eps)
      {
        return false;
      }
    }

    return true;
  }

  bool _IsExpressionInvalidDf(mi::base::Handle<const mi::neuraylib::IExpression> expr)
  {
    if (expr->get_kind() != mi::neuraylib::IExpression::Kind::EK_CONSTANT)
    {
      return false;
    }

    mi::base::Handle<const mi::neuraylib::IExpression_constant> constExpr(expr.get_interface<const mi::neuraylib::IExpression_constant>());
    mi::base::Handle<const mi::neuraylib::IValue> value(constExpr->get_value());

    return value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF;
  }

  bool _IsCompiledMaterialEmissive(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    mi::base::Handle<const mi::neuraylib::IExpression> emissionExpr(compiledMaterial->lookup_sub_expression("surface.emission.emission"));
    mi::base::Handle<const mi::neuraylib::IExpression> emissionIntensityExpr(compiledMaterial->lookup_sub_expression("surface.emission.intensity"));

    return !_IsExpressionInvalidDf(emissionExpr) && !_IsExpressionBlackColor(emissionIntensityExpr);
  }

  bool _IsCompiledMaterialThinWalled(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    mi::base::Handle<const mi::neuraylib::IExpression> expr(compiledMaterial->lookup_sub_expression("thin_walled"));

    if (expr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
    {
      return true;
    }

    mi::base::Handle<const mi::neuraylib::IExpression_constant> constExpr(expr->get_interface<const mi::neuraylib::IExpression_constant>());
    mi::base::Handle<const mi::neuraylib::IValue_bool> value(constExpr->get_value<mi::neuraylib::IValue_bool>());

    return !value || value->get_value();
  }

  bool _HasCompiledMaterialCutoutTransparency(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    float opacity = -1.0;
    return !compiledMaterial->get_cutout_opacity(&opacity) || opacity < 1.0f;
  }

  bool _HasCompiledMaterialBackfaceBsdf(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    mi::base::Handle<const mi::neuraylib::IExpression> expr(compiledMaterial->lookup_sub_expression("backface.scattering"));

    bool dfsDiffer = compiledMaterial->get_slot_hash(mi::neuraylib::SLOT_SURFACE_SCATTERING) != compiledMaterial->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_SCATTERING);

    return dfsDiffer && !_IsExpressionInvalidDf(expr);
  }

  bool _HasCompiledMaterialBackfaceEdf(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    mi::base::Handle<const mi::neuraylib::IExpression> expr(compiledMaterial->lookup_sub_expression("backface.emission.emission"));

    bool dfsDiffer = compiledMaterial->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_EDF_EMISSION) != compiledMaterial->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_EDF_EMISSION);

    return dfsDiffer && !_IsExpressionInvalidDf(expr);
  }

  bool _HasCompiledMaterialVolumeAbsorptionCoefficient(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    mi::base::Handle<const mi::neuraylib::IExpression> expr(compiledMaterial->lookup_sub_expression("volume.absorption_coefficient"));

    return !_IsExpressionBlackColor(expr);
  }

  bool _HasCompiledMaterialVolumeScatteringCoefficient(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    mi::base::Handle<const mi::neuraylib::IExpression> expr(compiledMaterial->lookup_sub_expression("volume.scattering_coefficient"));

    return !_IsExpressionBlackColor(expr);
  }

  // The MDL SDK can't generate code for the volume.scattering expression - in order to support anisotropy, we assume the common case.
  float _GetCompiledMaterialDirectionalBias(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    const float DEFAULT_BIAS = 0.0f; // isotropic

    mi::base::Handle<const mi::neuraylib::IExpression> expr(compiledMaterial->lookup_sub_expression("volume.scattering.directional_bias"));

    if (!expr || expr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
    {
      return DEFAULT_BIAS;
    }

    mi::base::Handle<const mi::neuraylib::IExpression_constant> constExpr(expr->get_interface<const mi::neuraylib::IExpression_constant>());
    mi::base::Handle<const mi::neuraylib::IValue_float> value(constExpr->get_value<mi::neuraylib::IValue_float>());

    if (!value)
    {
      return DEFAULT_BIAS;
    }

    return value->get_value();
  }

  std::vector<const char*> _ExtractSceneDataNames(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    std::vector<const char*> names;

    for (mi::Size i = 0; i < compiledMaterial->get_referenced_scene_data_count(); i++)
    {
      names.push_back(compiledMaterial->get_referenced_scene_data_name(i));
    }

    return names;
  }

  int _FindCameraPositionSceneDataIndex(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    for (mi::Size i = 0; i < compiledMaterial->get_referenced_scene_data_count(); i++)
    {
      const char* name = compiledMaterial->get_referenced_scene_data_name(i);

      if (strcmp(name, "CAMERA_POSITION") == 0)
      {
        return int(i) + 1; // index 0 is invalid
      }
    }
    return 0;
  }
}

namespace gtl
{
  McFrontend::McFrontend(const MaterialX::DocumentPtr mtlxStdLib,
                         McRuntime& runtime)
  {
    McMdlRuntime& mdlRuntime = runtime.getMdlRuntime();
    m_mdlMaterialCompiler = std::make_shared<McMdlMaterialCompiler>(mdlRuntime);
    m_mtlxMdlCodeGen = std::make_shared<McMtlxMdlCodeGen>(mtlxStdLib);
  }

  McMaterial* McFrontend::createFromMdlStr(std::string_view mdlSrc, std::string_view subIdentifier, bool hasCutoutTransparency)
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromString(mdlSrc, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    auto mdlMaterial = std::make_shared<McMdlMaterial>();
    mdlMaterial->compiledMaterial = compiledMaterial;

    return new McMaterial{
      .hasBackfaceBsdf = _HasCompiledMaterialBackfaceBsdf(compiledMaterial),
      .hasBackfaceEdf = _HasCompiledMaterialBackfaceEdf(compiledMaterial),
      .hasVolumeAbsorptionCoeff = _HasCompiledMaterialVolumeAbsorptionCoefficient(compiledMaterial),
      .hasVolumeScatteringCoeff = _HasCompiledMaterialVolumeScatteringCoefficient(compiledMaterial),
      .hasCutoutTransparency = hasCutoutTransparency,
      .isEmissive = _IsCompiledMaterialEmissive(compiledMaterial),
      .isThinWalled = _IsCompiledMaterialThinWalled(compiledMaterial),
      .directionalBias = _GetCompiledMaterialDirectionalBias(compiledMaterial),
      .resourcePathPrefix = "", // no source file
      .mdlMaterial = mdlMaterial,
      .requiresSceneTransforms = compiledMaterial->depends_on_state_transform(),
      .sceneDataNames = _ExtractSceneDataNames(compiledMaterial),
      .cameraPositionSceneDataIndex = _FindCameraPositionSceneDataIndex(compiledMaterial)
    };
  }

  McMaterial* McFrontend::createFromMtlxStr(std::string_view docStr)
  {
    std::string mdlSrc;
    std::string subIdentifier;
    bool hasCutoutTransparency;
    if (!m_mtlxMdlCodeGen->translate(docStr, mdlSrc, subIdentifier, hasCutoutTransparency))
    {
      return nullptr;
    }

    return createFromMdlStr(mdlSrc, subIdentifier, hasCutoutTransparency);
  }

  McMaterial* McFrontend::createFromMtlxDoc(const MaterialX::DocumentPtr doc)
  {
    std::string mdlSrc;
    std::string subIdentifier;
    bool hasCutoutTransparency;
    if (!m_mtlxMdlCodeGen->translate(doc, mdlSrc, subIdentifier, hasCutoutTransparency))
    {
      return nullptr;
    }

    return createFromMdlStr(mdlSrc, subIdentifier, hasCutoutTransparency);
  }

  McMaterial* McFrontend::createFromMdlFile(const char* filePath, std::string_view subIdentifier)
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

    return new McMaterial{
      .hasBackfaceBsdf = _HasCompiledMaterialBackfaceBsdf(compiledMaterial),
      .hasBackfaceEdf = _HasCompiledMaterialBackfaceEdf(compiledMaterial),
      .hasVolumeAbsorptionCoeff = _HasCompiledMaterialVolumeAbsorptionCoefficient(compiledMaterial),
      .hasVolumeScatteringCoeff = _HasCompiledMaterialVolumeScatteringCoefficient(compiledMaterial),
      .hasCutoutTransparency = _HasCompiledMaterialCutoutTransparency(compiledMaterial),
      .isEmissive = _IsCompiledMaterialEmissive(compiledMaterial),
      .isThinWalled = _IsCompiledMaterialThinWalled(compiledMaterial),
      .directionalBias = _GetCompiledMaterialDirectionalBias(compiledMaterial),
      .resourcePathPrefix = resourcePathPrefix,
      .mdlMaterial = mdlMaterial,
      .requiresSceneTransforms = compiledMaterial->depends_on_state_transform(),
      .sceneDataNames = _ExtractSceneDataNames(compiledMaterial),
      .cameraPositionSceneDataIndex = _FindCameraPositionSceneDataIndex(compiledMaterial)
    };
  }
}
