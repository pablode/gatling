#include "shadergen.h"

#include "GlslangShaderCompiler.h"
#include "MtlxMdlTranslator.h"

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>

struct SgMaterial
{
  std::string mdlSrc;
  std::string subIdentifier;
};

std::string s_shaderPath;
std::unique_ptr<sg::IShaderCompiler> s_shaderCompiler;
std::unique_ptr<sg::MtlxMdlTranslator> s_mtlxMdlTranslator;

bool sgInitialize(const char* shaderPath,
                  const char* mtlxlibPath)
{
  s_shaderPath = shaderPath;
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
  mat->mdlSrc = mdlSrc;
  mat->subIdentifier = subIdentifier;
  return mat;
}

void sgDestroyMaterial(SgMaterial* mat)
{
  delete mat;
}

template<typename T>
constexpr const char* _sgGetHlslType(T var)
{
  if constexpr(std::is_same<T, uint32_t>::value)
  {
    return "uint";
  }
  else if constexpr(std::is_same<T, float>::value)
  {
    return "float";
  }
  else
  {
    static_assert(false, "HLSL type mapping does not exist");
  }
}

bool sgGenerateMainShader(const SgMainShaderParams* params,
                          uint32_t* spvSize,
                          uint32_t** spv,
                          const char** entryPoint)
{
  std::stringstream ss;
  ss << std::showpoint;
  ss << std::setprecision(std::numeric_limits<float>::digits10);

#define APPEND_CONSTANT(name, cvar)     \
  ss << "const ";                       \
  ss << _sgGetHlslType(params->cvar);   \
  ss << " " << name << " = ";           \
  ss << params->cvar << ";\n";

  APPEND_CONSTANT("NUM_THREADS_X", num_threads_x)
  APPEND_CONSTANT("NUM_THREADS_Y", num_threads_y)
  APPEND_CONSTANT("MAX_STACK_SIZE", max_stack_size)
  APPEND_CONSTANT("SAMPLE_COUNT", spp)
  APPEND_CONSTANT("MAX_BOUNCES", max_bounces)
  APPEND_CONSTANT("RR_BOUNCE_OFFSET", rr_bounce_offset)
  APPEND_CONSTANT("RR_INV_MIN_TERM_PROB", rr_inv_min_term_prob)

  std::string fileName = "main.comp.hlsl";
  std::string filePath = s_shaderPath + "/" + fileName;
  *entryPoint = "CSMain";

  std::string source = ss.str() + "#include \"" + fileName + "\"\n";

  return s_shaderCompiler->compileHlslToSpv(source, filePath, *entryPoint, spvSize, spv);
}
