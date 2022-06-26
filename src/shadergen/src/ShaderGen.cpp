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

#include "ShaderGen.h"

#include "MtlxMdlCodeGen.h"
#include "MdlRuntime.h"
#include "MdlMaterialCompiler.h"
#include "MdlHlslCodeGen.h"

#ifdef GATLING_USE_GLSLANG
#include "GlslangShaderCompiler.h"
#else
#include "DxcShaderCompiler.h"
#endif

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <fstream>
#include <cassert>

namespace sg
{
  struct Material
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    bool isEmissive;
  };

  bool ShaderGen::init(const InitParams& params)
  {
    m_shaderPath = std::string(params.shaderPath);

    m_mdlRuntime = new sg::MdlRuntime();
    if (!m_mdlRuntime->init(params.resourcePath.data()))
    {
      return false;
    }

    m_mdlHlslCodeGen = new sg::MdlHlslCodeGen();
    if (!m_mdlHlslCodeGen->init(*m_mdlRuntime))
    {
      return false;
    }

    m_mdlMaterialCompiler = new sg::MdlMaterialCompiler(*m_mdlRuntime, params.mdlLibPath.data());

  #ifdef GATLING_USE_GLSLANG
    m_shaderCompiler = new sg::GlslangShaderCompiler(m_shaderPath);
  #else
    m_shaderCompiler = new sg::DxcShaderCompiler(m_shaderPath);
  #endif
    if (!m_shaderCompiler->init())
    {
      return false;
    }

    m_mtlxMdlCodeGen = new sg::MtlxMdlCodeGen(params.mtlxLibPath.data());

    return true;
  }

  ShaderGen::~ShaderGen()
  {
    delete m_mtlxMdlCodeGen;
    delete m_shaderCompiler;
    delete m_mdlMaterialCompiler;
    delete m_mdlHlslCodeGen;
    delete m_mdlRuntime;
  }

  bool _sgIsMaterialEmissive(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
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

    return v0->get_value() != 0.0f || v1->get_value() != 0.0f || v2->get_value() != 0.0f;
  }

  Material* ShaderGen::createMaterialFromMtlx(std::string_view docStr)
  {
    std::string mdlSrc;
    std::string subIdentifier;
    if (!m_mtlxMdlCodeGen->translate(docStr, mdlSrc, subIdentifier))
    {
      return nullptr;
    }

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromString(mdlSrc, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    Material* mat = new Material();
    mat->compiledMaterial = compiledMaterial;
    mat->isEmissive = _sgIsMaterialEmissive(compiledMaterial);
    return mat;
  }

  Material* ShaderGen::createMaterialFromMdlFile(std::string_view filePath, std::string_view subIdentifier)
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromFile(filePath, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    Material* mat = new Material();
    mat->compiledMaterial = compiledMaterial;
    return mat;
  }

  void ShaderGen::destroyMaterial(Material* mat)
  {
    delete mat;
  }

  bool ShaderGen::isMaterialEmissive(const struct Material* mat)
  {
    return mat->isEmissive;
  }

  bool _sgReadTextFromFile(const std::string& filePath, std::string& text)
  {
    std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
    if (!file.is_open())
    {
      return false;
    }
    file.seekg(0, std::ios_base::end);
    text.resize(file.tellg(), ' ');
    file.seekg(0, std::ios_base::beg);
    file.read(&text[0], text.size());
    return file.good();
  }

  bool ShaderGen::generateMainShader(const MainShaderParams* params,
                                     MainShaderResult& result)
  {
    std::string fileName = "main.comp.hlsl";
    std::string filePath = m_shaderPath + "/" + fileName;

    std::vector<const mi::neuraylib::ICompiled_material*> compiledMaterials;
    for (uint32_t i = 0; i < params->materials.size(); i++)
    {
      compiledMaterials.push_back(params->materials[i]->compiledMaterial.get());
    }

    std::string genMdl;
    if (!m_mdlHlslCodeGen->translate(compiledMaterials, genMdl, result.textureResources))
    {
      return false;
    }

    std::string fileSrc;
    if (!_sgReadTextFromFile(filePath, fileSrc))
    {
      return false;
    }

    std::string mdlLocMarker = "MDL_GENERATED_CODE";
    size_t mdlInjectionLoc = fileSrc.find(mdlLocMarker);
    assert(mdlInjectionLoc != std::string::npos);
    fileSrc.replace(mdlInjectionLoc, mdlLocMarker.size(), genMdl);

    std::stringstream ss;
    ss << std::showpoint;
    ss << std::setprecision(std::numeric_limits<float>::digits10);

    int textureCount = result.textureResources.size();
    if (textureCount > 0)
    {
      ss << "#define HAS_TEXTURES_2D\n";
      ss << "#define TEXTURE_COUNT_2D " << textureCount << "\n";
    }

#if !defined(NDEBUG) && !defined(__APPLE__)
    ss << "#define DEBUG_PRINTF\n";
#endif

#define APPEND_CONSTANT(name, cvar) \
    ss << "#define " << name << " " << params->cvar << "\n";
#define APPEND_DEFINE(name, cvar) \
    if (params->cvar) ss << "#define " << name << "\n";

    APPEND_CONSTANT("AOV_ID", aovId)
    APPEND_CONSTANT("NUM_THREADS_X", numThreadsX)
    APPEND_CONSTANT("NUM_THREADS_Y", numThreadsY)
    APPEND_CONSTANT("MAX_STACK_SIZE", maxStackSize)
    APPEND_CONSTANT("POSTPONE_RATIO", postponeRatio)
    APPEND_CONSTANT("FACE_COUNT", faceCount)
    APPEND_CONSTANT("EMISSIVE_FACE_COUNT", emissiveFaceCount)
    APPEND_DEFINE("BVH_ENABLED", bvh)
    APPEND_DEFINE("TRIANGLE_POSTPONING", trianglePostponing)
    APPEND_DEFINE("NEXT_EVENT_ESTIMATION", nextEventEstimation)

    ss << fileSrc;

    result.entryPoint = "CSMain";
    return m_shaderCompiler->compileHlslToSpv(ss.str(), filePath, result.entryPoint, result.spv);
  }
}
