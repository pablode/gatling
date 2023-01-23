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
#include "MdlGlslCodeGen.h"
#include "GlslangShaderCompiler.h"

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <fstream>
#include <cassert>

namespace gi::sg
{
  struct Material
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    bool isEmissive;
    bool isOpaque;
  };

  bool ShaderGen::init(const InitParams& params)
  {
    m_shaderPath = std::string(params.shaderPath);

    m_mdlRuntime = new sg::MdlRuntime();
    if (!m_mdlRuntime->init(params.resourcePath.data()))
    {
      return false;
    }

    m_mdlGlslCodeGen = new sg::MdlGlslCodeGen();
    if (!m_mdlGlslCodeGen->init(*m_mdlRuntime))
    {
      return false;
    }

    m_mdlMaterialCompiler = new sg::MdlMaterialCompiler(*m_mdlRuntime, params.mdlLibPath.data());

    m_mtlxMdlCodeGen = new sg::MtlxMdlCodeGen(params.mtlxLibPath.data());

    if (!sg::GlslangShaderCompiler::init())
    {
      return false;
    }
    m_shaderCompiler = new sg::GlslangShaderCompiler(m_shaderPath);

    return true;
  }

  ShaderGen::~ShaderGen()
  {
    sg::GlslangShaderCompiler::deinit();
    delete m_mtlxMdlCodeGen;
    delete m_shaderCompiler;
    delete m_mdlMaterialCompiler;
    delete m_mdlGlslCodeGen;
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

  bool _sgIsMaterialOpaque(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    float opacity = -1.0f;

    return compiledMaterial->get_cutout_opacity(&opacity) && opacity >= 1.0f;
  }

  Material* _sgMakeMaterial(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    Material* m = new Material();
    m->compiledMaterial = compiledMaterial;
    m->isEmissive = _sgIsMaterialEmissive(compiledMaterial);
    m->isOpaque = _sgIsMaterialOpaque(compiledMaterial);
    return m;
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

    return _sgMakeMaterial(compiledMaterial);
  }

  Material* ShaderGen::createMaterialFromMdlFile(std::string_view filePath, std::string_view subIdentifier)
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromFile(filePath, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    return _sgMakeMaterial(compiledMaterial);
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
    std::string fileName = "main.comp.glsl";
    std::string filePath = m_shaderPath + "/" + fileName;

    std::vector<const mi::neuraylib::ICompiled_material*> compiledMaterials;
    for (uint32_t i = 0; i < params->materials.size(); i++)
    {
      compiledMaterials.push_back(params->materials[i]->compiledMaterial.get());
    }

    std::string genMdl;
    if (!m_mdlGlslCodeGen->translate(compiledMaterials, genMdl, result.textureResources))
    {
      return false;
    }

    std::string fileSrc;
    if (!_sgReadTextFromFile(filePath, fileSrc))
    {
      return false;
    }

    std::stringstream ss;
    ss << std::showpoint;
    ss << std::setprecision(std::numeric_limits<float>::digits10);

    ss << "#version 460 core\n";

    // FIXME: unfortunately we can't enable #extension requirements using the GLSL preprocessor..
    if (params->shaderClockExts)
    {
      ss << "#extension GL_EXT_shader_explicit_arithmetic_types_int64: require\n";
      ss << "#extension GL_ARB_shader_clock: require\n";
    }

    // Remove MDL struct definitions because they're too bloated. We know more about the
    // data from which the code is generated from and can reduce the memory footprint.
    size_t mdlCodeOffset = genMdl.find("// user defined structs");
    assert(mdlCodeOffset != std::string::npos);
    genMdl = genMdl.substr(mdlCodeOffset, genMdl.size() - mdlCodeOffset);

    std::string mdlLocMarker = "#pragma MDL_GENERATED_CODE";
    size_t mdlInjectionLoc = fileSrc.find(mdlLocMarker);
    assert(mdlInjectionLoc != std::string::npos);
    fileSrc.replace(mdlInjectionLoc, mdlLocMarker.size(), genMdl);

    int textureCount2d = 0;
    int textureCount3d = 0;
    for (auto& texResource : result.textureResources)
    {
      (texResource.is3dImage ? textureCount3d : textureCount2d)++;
    }
    if (textureCount2d > 0)
    {
      ss << "#define HAS_TEXTURES_2D\n";
      ss << "#define TEXTURE_COUNT_2D " << textureCount2d << "\n";
    }
    if (textureCount3d > 0)
    {
      ss << "#define HAS_TEXTURES_3D\n";
      ss << "#define TEXTURE_COUNT_3D " << textureCount3d << "\n";
    }

#if defined(NDEBUG) || defined(__APPLE__)
    ss << "#define NDEBUG\n";
#endif

#define APPEND_CONSTANT(name, cvar) \
    ss << "#define " << name << " " << params->cvar << "\n";
#define APPEND_DEFINE(name, cvar) \
    if (params->cvar) ss << "#define " << name << "\n";

    APPEND_CONSTANT("AOV_ID", aovId)
    APPEND_CONSTANT("NUM_THREADS_X", numThreadsX)
    APPEND_CONSTANT("NUM_THREADS_Y", numThreadsY)
    APPEND_CONSTANT("FACE_COUNT", faceCount)
    APPEND_CONSTANT("EMISSIVE_FACE_COUNT", emissiveFaceCount)
    APPEND_DEFINE("TRIANGLE_POSTPONING", trianglePostponing)
    APPEND_DEFINE("NEXT_EVENT_ESTIMATION", nextEventEstimation)

    ss << fileSrc;

    std::string glslStr = ss.str();
    if (getenv("GATLING_DUMP_GLSL"))
    {
      printf("GLSL source: %s\n", glslStr.c_str());
    }

    return m_shaderCompiler->compileGlslToSpv(GlslangShaderCompiler::ShaderStage::Compute, glslStr, filePath, result.spv);
  }
}