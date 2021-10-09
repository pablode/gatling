#include "shadergen.h"

#include "GlslangShaderCompiler.h"
#include "MtlxMdlTranslator.h"
#include "MdlHlslTranslator.h"
#include "MdlRuntime.h"

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>

struct SgMaterial
{
  sg::SourceIdentifierPair srcAndIdentifier;
};

std::string s_shaderPath;

std::unique_ptr<sg::MdlRuntime> s_mdlRuntime;
std::unique_ptr<sg::MdlHlslTranslator> s_mdlHlslTranslator;
std::unique_ptr<sg::MtlxMdlTranslator> s_mtlxMdlTranslator;
std::unique_ptr<sg::IShaderCompiler> s_shaderCompiler;

bool sgInitialize(const char* resourcePath,
                  const char* shaderPath,
                  const char* mtlxlibPath,
                  const char* mtlxmdlPath)
{
  s_shaderPath = shaderPath;

  s_mdlRuntime = std::make_unique<sg::MdlRuntime>();
  if (!s_mdlRuntime->init(resourcePath))
  {
    return false;
  }

  auto& neuray = s_mdlRuntime->getNeuray();

  s_mdlHlslTranslator = std::make_unique<sg::MdlHlslTranslator>();
  if (!s_mdlHlslTranslator->init(neuray, mtlxmdlPath))
  {
    return false;
  }

  s_shaderCompiler = std::make_unique<sg::GlslangShaderCompiler>(s_shaderPath);
  if (!s_shaderCompiler->init())
  {
    return false;
  }

  s_mtlxMdlTranslator = std::make_unique<sg::MtlxMdlTranslator>(mtlxlibPath);

  return true;
}

void sgTerminate()
{
  s_mdlHlslTranslator.reset();
  s_mtlxMdlTranslator.reset();
  s_shaderCompiler.reset();
  s_mdlRuntime.reset();
}

SgMaterial* sgCreateMaterialFromMtlx(const char* docStr)
{
  std::string mdlSrc;
  std::string subIdentifier;
  if (!s_mtlxMdlTranslator->translate(docStr, mdlSrc, subIdentifier))
  {
    return nullptr;
  }

  SgMaterial* mat = new SgMaterial();

  sg::SourceIdentifierPair& pair = mat->srcAndIdentifier;
  pair.src = mdlSrc;
  pair.identifier = subIdentifier;
  return mat;
}

void sgDestroyMaterial(SgMaterial* mat)
{
  delete mat;
}

bool _sgGenerateMainShaderMdlCode(uint32_t materialCount,
                                  const struct SgMaterial** materials,
                                  std::string& generatedHlsl)
{
  std::vector<const sg::SourceIdentifierPair*> srcIdVec;

  for (uint32_t i = 0; i < materialCount; i++)
  {
    const sg::SourceIdentifierPair& pair = materials[i]->srcAndIdentifier;
    srcIdVec.push_back(&pair);
  }

  return s_mdlHlslTranslator->translate(srcIdVec, generatedHlsl);
}

bool sgGenerateMainShader(const SgMainShaderParams* params,
                          uint32_t* spvSize,
                          uint32_t** spv,
                          const char** entryPoint)
{
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
  APPEND_CONSTANT("RR_BOUNCE_OFFSET", rr_bounce_offset)
  APPEND_CONSTANT("RR_INV_MIN_TERM_PROB", rr_inv_min_term_prob)

  std::string genMdlHlsl;
  if (!_sgGenerateMainShaderMdlCode(params->material_count,
                                    params->materials,
                                    genMdlHlsl))
  {
    return false;
  }

  std::string fileName = "main.comp.hlsl";
  std::string filePath = s_shaderPath + "/" + fileName;

  ss << "#include \"mdl_types.hlsl\"\n";
  ss << genMdlHlsl;
  ss << "#include \"" + fileName + "\"\n";

  *entryPoint = "CSMain";
  return s_shaderCompiler->compileHlslToSpv(ss.str(), filePath, *entryPoint, spvSize, spv);
}
