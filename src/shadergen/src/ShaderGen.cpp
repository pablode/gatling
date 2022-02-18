#include "ShaderGen.h"

#include "MtlxMdlCodeGen.h"
#include "MdlRuntime.h"
#include "MdlMaterialCompiler.h"
#include "MdlHlslCodeGen.h"

#ifdef GATLING_USE_DXC
#include "DxcShaderCompiler.h"
#else
#include "GlslangShaderCompiler.h"
#endif

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <fstream>

namespace sg
{
  struct Material
  {
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
  };

  bool ShaderGen::init(const InitParams& params)
  {
    m_shaderPath = std::string(params.shaderPath);

    m_mdlRuntime = new sg::MdlRuntime();
    if (!m_mdlRuntime->init(params.resourcePath.data(), params.mtlxmdlPath.data()))
    {
      return false;
    }

    m_mdlHlslCodeGen = new sg::MdlHlslCodeGen();
    if (!m_mdlHlslCodeGen->init(*m_mdlRuntime))
    {
      return false;
    }

    m_mdlMaterialCompiler = new sg::MdlMaterialCompiler(*m_mdlRuntime);

  #ifdef GATLING_USE_DXC
    m_shaderCompiler = new sg::DxcShaderCompiler(m_shaderPath);
  #else
    m_shaderCompiler = new sg::GlslangShaderCompiler(m_shaderPath);
  #endif
    if (!m_shaderCompiler->init())
    {
      return false;
    }

    m_mtlxMdlCodeGen = new sg::MtlxMdlCodeGen(params.mtlxlibPath.data());

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

  Material* ShaderGen::createMaterialFromMtlx(std::string_view docStr)
  {
    std::string mdlSrc;
    std::string subIdentifier;
    if (!m_mtlxMdlCodeGen->translate(docStr, mdlSrc, subIdentifier))
    {
      return nullptr;
    }

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    if (!m_mdlMaterialCompiler->compileMaterial(mdlSrc, subIdentifier, compiledMaterial))
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

  bool _sgGenerateMainShaderMdlHlsl(MdlHlslCodeGen& codeGen,
                                    const std::vector<Material*>& materials,
                                    std::string& hlsl)
  {
    std::vector<const mi::neuraylib::ICompiled_material*> compiledMaterials;

    for (uint32_t i = 0; i < materials.size(); i++)
    {
      compiledMaterials.push_back(materials[i]->compiledMaterial.get());
    }

    return codeGen.translate(compiledMaterials, hlsl);
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
                                     std::vector<uint8_t>& spv,
                                     std::string& entryPoint)
  {
    std::string fileName = "main.comp.hlsl";
    std::string filePath = m_shaderPath + "/" + fileName;

    std::stringstream ss;
    ss << std::showpoint;
    ss << std::setprecision(std::numeric_limits<float>::digits10);

#if !defined(NDEBUG) && !defined(__APPLE__)
    ss << "#define DEBUG_PRINTF\n";
#endif

  #define APPEND_CONSTANT(name, cvar) \
    ss << "#define " << name << " " << params->cvar << "\n";

    APPEND_CONSTANT("NUM_THREADS_X", numThreadsX)
    APPEND_CONSTANT("NUM_THREADS_Y", numThreadsY)
    APPEND_CONSTANT("MAX_STACK_SIZE", maxStackSize)
    APPEND_CONSTANT("POSTPONE_RATIO", postponeRatio)

    std::string genMdl;
    if (!_sgGenerateMainShaderMdlHlsl(*m_mdlHlslCodeGen, params->materials, genMdl))
    {
      return false;
    }

    std::string fileSrc;
    if (!_sgReadTextFromFile(filePath, fileSrc))
    {
      return false;
    }

    ss << "#include \"mdl_types.hlsl\"\n";
    ss << genMdl;
    ss << fileSrc;

    entryPoint = "CSMain";
    return m_shaderCompiler->compileHlslToSpv(ss.str(), filePath, entryPoint, spv);
  }
}
