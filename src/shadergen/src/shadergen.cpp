#include "shadergen.h"

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

struct SgMaterial
{
  mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
};

std::string s_shaderPath;

std::unique_ptr<sg::MdlRuntime> s_mdlRuntime;
std::unique_ptr<sg::MdlMaterialCompiler> s_mdlMaterialCompiler;
std::unique_ptr<sg::MdlHlslCodeGen> s_mdlHlslCodeGen;
std::unique_ptr<sg::MtlxMdlCodeGen> s_mtlxMdlCodeGen;
std::unique_ptr<sg::IShaderCompiler> s_shaderCompiler;

bool sgInitialize(const char* resourcePath,
                  const char* shaderPath,
                  const char* mtlxlibPath,
                  const char* mtlxmdlPath)
{
  s_shaderPath = shaderPath;

  s_mdlRuntime = std::make_unique<sg::MdlRuntime>();
  if (!s_mdlRuntime->init(resourcePath, mtlxmdlPath))
  {
    return false;
  }

  s_mdlHlslCodeGen = std::make_unique<sg::MdlHlslCodeGen>();
  if (!s_mdlHlslCodeGen->init(*s_mdlRuntime))
  {
    return false;
  }

  s_mdlMaterialCompiler = std::make_unique<sg::MdlMaterialCompiler>(*s_mdlRuntime);

#ifdef GATLING_USE_DXC
  s_shaderCompiler = std::make_unique<sg::DxcShaderCompiler>(s_shaderPath);
#else
  s_shaderCompiler = std::make_unique<sg::GlslangShaderCompiler>(s_shaderPath);
#endif
  if (!s_shaderCompiler->init())
  {
    return false;
  }

  s_mtlxMdlCodeGen = std::make_unique<sg::MtlxMdlCodeGen>(mtlxlibPath);

  return true;
}

void sgTerminate()
{
  s_mtlxMdlCodeGen.reset();
  s_shaderCompiler.reset();
  s_mdlMaterialCompiler.reset();
  s_mdlHlslCodeGen.reset();
  s_mdlRuntime.reset();
}

SgMaterial* sgCreateMaterialFromMtlx(const char* docStr)
{
  std::string mdlSrc;
  std::string subIdentifier;
  if (!s_mtlxMdlCodeGen->translate(docStr, mdlSrc, subIdentifier))
  {
    return nullptr;
  }

  mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
  if (!s_mdlMaterialCompiler->compileMaterial(mdlSrc, subIdentifier, compiledMaterial))
  {
    return nullptr;
  }

  SgMaterial* mat = new SgMaterial();
  mat->compiledMaterial = compiledMaterial;
  return mat;
}

void sgDestroyMaterial(SgMaterial* mat)
{
  delete mat;
}

bool _sgGenerateMainShaderMdlHlsl(uint32_t materialCount,
                                  const struct SgMaterial** materials,
                                  std::string& hlsl)
{
  std::vector<const mi::neuraylib::ICompiled_material*> compiledMaterials;

  for (uint32_t i = 0; i < materialCount; i++)
  {
    compiledMaterials.push_back(materials[i]->compiledMaterial.get());
  }

  return s_mdlHlslCodeGen->translate(compiledMaterials, hlsl);
}

bool _sgReadTextFromFile(const std::string& filePath,
                         std::string& text)
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

bool sgGenerateMainShader(const SgMainShaderParams* params,
                          uint32_t* spvSize,
                          uint8_t** spv,
                          const char** entryPoint)
{
  std::string fileName = "main.comp.hlsl";
  std::string filePath = s_shaderPath + "/" + fileName;

  std::stringstream ss;
  ss << std::showpoint;
  ss << std::setprecision(std::numeric_limits<float>::digits10);

#define APPEND_CONSTANT(name, cvar) \
  ss << "#define " << name << " " << params->cvar << "\n";

  APPEND_CONSTANT("NUM_THREADS_X", num_threads_x)
  APPEND_CONSTANT("NUM_THREADS_Y", num_threads_y)
  APPEND_CONSTANT("MAX_STACK_SIZE", max_stack_size)
  APPEND_CONSTANT("SAMPLE_COUNT", spp)
  APPEND_CONSTANT("MAX_BOUNCES", max_bounces)
  APPEND_CONSTANT("MAX_SAMPLE_VALUE", max_sample_value)
  APPEND_CONSTANT("RR_BOUNCE_OFFSET", rr_bounce_offset)
  APPEND_CONSTANT("RR_INV_MIN_TERM_PROB", rr_inv_min_term_prob)

  ss << "#define BACKGROUND_COLOR float4("
      << params->bg_color[0] << ", "
      << params->bg_color[1] << ", "
      << params->bg_color[2] << ", "
      << params->bg_color[3] << ")\n";

  std::string genMdl;
  if (!_sgGenerateMainShaderMdlHlsl(params->material_count,
                                    params->materials,
                                    genMdl))
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

  *entryPoint = "CSMain";
  return s_shaderCompiler->compileHlslToSpv(ss.str(), filePath, *entryPoint, spvSize, spv);
}
