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
#include "GlslSourceStitcher.h"

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
    std::string resourcePathPrefix;
  };

  bool ShaderGen::init(const InitParams& params)
  {
    m_shaderPath = fs::path(params.shaderPath);

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

    m_mdlMaterialCompiler = new sg::MdlMaterialCompiler(*m_mdlRuntime, params.mdlSearchPaths);

    m_mtlxMdlCodeGen = new sg::MtlxMdlCodeGen(params.mtlxSearchPaths);

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

    const float eps = 1e-7f;
    return v0->get_value() > eps || v1->get_value() > eps || v2->get_value() > eps;
  }

  bool _sgIsMaterialOpaque(mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial)
  {
    return compiledMaterial->get_opacity() == mi::neuraylib::OPACITY_OPAQUE;
  }

  Material* ShaderGen::createMaterialFromMtlxStr(std::string_view docStr)
  {
    std::string mdlSrc;
    std::string subIdentifier;
    bool isOpaque;
    if (!m_mtlxMdlCodeGen->translate(docStr, mdlSrc, subIdentifier, isOpaque))
    {
      return nullptr;
    }

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromString(mdlSrc, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    Material* m = new Material();
    m->compiledMaterial = compiledMaterial;
    m->isEmissive = _sgIsMaterialEmissive(compiledMaterial);
    m->isOpaque = isOpaque;
    return m;
  }

  Material* ShaderGen::createMaterialFromMtlxDoc(const MaterialX::DocumentPtr doc)
  {
    // FIXME: deduplicate code
    std::string mdlSrc;
    std::string subIdentifier;
    bool isOpaque;
    if (!m_mtlxMdlCodeGen->translate(doc, mdlSrc, subIdentifier, isOpaque))
    {
      return nullptr;
    }

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromString(mdlSrc, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    Material* m = new Material();
    m->compiledMaterial = compiledMaterial;
    m->isEmissive = _sgIsMaterialEmissive(compiledMaterial);
    m->isOpaque = isOpaque;
    return m;
  }

  Material* ShaderGen::createMaterialFromMdlFile(std::string_view filePath, std::string_view subIdentifier)
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileFromFile(filePath, subIdentifier, compiledMaterial))
    {
      return nullptr;
    }

    std::string resourcePathPrefix = fs::path(filePath).parent_path().string();

    Material* m = new Material();
    m->compiledMaterial = compiledMaterial;
    m->isEmissive = _sgIsMaterialEmissive(compiledMaterial);
    m->isOpaque = _sgIsMaterialOpaque(compiledMaterial);
    m->resourcePathPrefix = resourcePathPrefix;
    return m;
  }

  void ShaderGen::destroyMaterial(Material* mat)
  {
    delete mat;
  }

  bool ShaderGen::isMaterialEmissive(const Material* mat)
  {
    return mat->isEmissive;
  }

  bool ShaderGen::isMaterialOpaque(const Material* mat)
  {
    return mat->isOpaque;
  }

  void _sgGenerateCommonDefines(GlslSourceStitcher& stitcher, uint32_t texCount2d, uint32_t texCount3d)
  {
#if defined(NDEBUG) || defined(__APPLE__)
    stitcher.appendDefine("NDEBUG");
#endif

    if (texCount2d > 0)
    {
      stitcher.appendDefine("HAS_TEXTURES_2D");
      stitcher.appendDefine("TEXTURE_COUNT_2D", texCount2d);
    }
    if (texCount3d > 0)
    {
      stitcher.appendDefine("HAS_TEXTURES_3D");
      stitcher.appendDefine("TEXTURE_COUNT_3D", texCount3d);
    }
  }

  bool ShaderGen::generateRgenSpirv(std::string_view fileName, const RaygenShaderParams& params, std::vector<uint8_t>& spv)
  {
    GlslSourceStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.texCount2d, params.texCount3d);

    // FIXME: 'enable' instead?
    if (params.shaderClockExts)
    {
      stitcher.appendRequiredExtension("GL_EXT_shader_explicit_arithmetic_types_int64");
      stitcher.appendRequiredExtension("GL_ARB_shader_clock");
    }

    stitcher.appendDefine("AOV_ID", params.aovId);

    fs::path filePath = m_shaderPath / fileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GlslangShaderCompiler::ShaderStage::RayGen, source, spv);
  }

  bool ShaderGen::generateMissSpirv(std::string_view fileName, const MissShaderParams& params, std::vector<uint8_t>& spv)
  {
    GlslSourceStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.texCount2d, params.texCount3d);

    if (params.domeLightEnabled)
    {
      stitcher.appendDefine("DOMELIGHT_ENABLED");
    }

    fs::path filePath = m_shaderPath / fileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GlslangShaderCompiler::ShaderStage::Miss, source, spv);
  }

  bool _genInfoFromCodeGenResult(const MdlGlslCodeGenResult& codeGenResult,
                                 const std::string& resourcePathPrefix,
                                 fs::path shaderPath,
                                 ShaderGen::MaterialGlslGenInfo& genInfo)
  {
    // Append resource path prefix for file-backed MDL modules.
    genInfo.textureResources = codeGenResult.textureResources;

    if (!resourcePathPrefix.empty())
    {
      for (sg::TextureResource& texRes : genInfo.textureResources)
      {
        texRes.filePath = resourcePathPrefix + texRes.filePath;
      }
    }

    // Remove MDL struct definitions because they're too bloated. We know more about the
    // data from which the code is generated from and can reduce the memory footprint.
    std::string glslSource = codeGenResult.glslSource;
    size_t mdlCodeOffset = glslSource.find("// user defined structs");
    assert(mdlCodeOffset != std::string::npos);
    glslSource = glslSource.substr(mdlCodeOffset, glslSource.size() - mdlCodeOffset);

    GlslSourceStitcher stitcher;
    if (!stitcher.appendSourceFile(shaderPath / "mdl_types.glsl"))
    {
      return false;
    }
    if (!stitcher.appendSourceFile(shaderPath / "mdl_interface.glsl"))
    {
      return false;
    }
    stitcher.appendString(glslSource);

    genInfo.glslSource = stitcher.source();

    return true;
  }

  bool ShaderGen::generateMaterialShadingGenInfo(const Material* material, MaterialGlslGenInfo& genInfo)
  {
    const mi::neuraylib::ICompiled_material* compiledMaterial = material->compiledMaterial.get();

    MdlGlslCodeGenResult codeGenResult;
    if (!m_mdlGlslCodeGen->genMaterialShadingCode(compiledMaterial, codeGenResult))
    {
      return false;
    }

    return _genInfoFromCodeGenResult(codeGenResult, material->resourcePathPrefix, m_shaderPath, genInfo);
  }

  bool ShaderGen::generateMaterialOpacityGenInfo(const Material* material, MaterialGlslGenInfo& genInfo)
  {
    const mi::neuraylib::ICompiled_material* compiledMaterial = material->compiledMaterial.get();

    MdlGlslCodeGenResult codeGenResult;
    if (!m_mdlGlslCodeGen->genMaterialOpacityCode(compiledMaterial, codeGenResult))
    {
      return false;
    }

    return _genInfoFromCodeGenResult(codeGenResult, material->resourcePathPrefix, m_shaderPath, genInfo);
  }

  bool ShaderGen::generateClosestHitSpirv(const ClosestHitShaderParams& params, std::vector<uint8_t>& spv)
  {
    GlslSourceStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.texCount2d, params.texCount3d);

    stitcher.appendDefine("AOV_ID", params.aovId);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_2D", params.textureIndexOffset2d);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_3D", params.textureIndexOffset3d);
    if (params.isOpaque)
    {
      stitcher.appendDefine("IS_OPAQUE", params.aovId);
    }

    fs::path filePath = m_shaderPath / params.baseFileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    stitcher.replaceFirst("#pragma MDL_GENERATED_CODE", params.shadingGlsl);

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GlslangShaderCompiler::ShaderStage::ClosestHit, source, spv);
  }

  bool ShaderGen::generateAnyHitSpirv(const AnyHitShaderParams& params, std::vector<uint8_t>& spv)
  {
    GlslSourceStitcher stitcher;
    stitcher.appendVersion();

    _sgGenerateCommonDefines(stitcher, params.texCount2d, params.texCount3d);

    stitcher.appendDefine("AOV_ID", params.aovId);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_2D", params.textureIndexOffset2d);
    stitcher.appendDefine("TEXTURE_INDEX_OFFSET_3D", params.textureIndexOffset3d);
    if (params.shadowTest)
    {
      stitcher.appendDefine("SHADOW_TEST");
    }

    fs::path filePath = m_shaderPath / params.baseFileName;
    if (!stitcher.appendSourceFile(filePath))
    {
      return false;
    }

    stitcher.replaceFirst("#pragma MDL_GENERATED_CODE", params.opacityEvalGlsl);

    std::string source = stitcher.source();
    return m_shaderCompiler->compileGlslToSpv(GlslangShaderCompiler::ShaderStage::AnyHit, source, spv);
  }
}
