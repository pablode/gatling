#include "shadergen.h"

#include "GlslangShaderCompiler.h"

#include <string>
#include <sstream>
#include <limits>
#include <iomanip>

std::string s_shaderPath;
std::unique_ptr<sg::IShaderCompiler> s_shaderCompiler;

bool sgInitialize(const char* resourcePath)
{
  s_shaderPath = std::string(resourcePath) + "/shaders";
  s_shaderCompiler = std::make_unique<sg::GlslangShaderCompiler>(s_shaderPath);

  if (!s_shaderCompiler->init())
  {
    return false;
  }

  return true;
}

void sgTerminate()
{
}

bool sgGenerateMainShader(const SgMainShaderParams* params,
                          uint32_t* spvSize,
                          uint32_t** spv,
                          const char** entryPoint)
{
  std::stringstream ss;
  ss << std::showpoint;
  ss << std::setprecision(std::numeric_limits<float>::digits10);

#define HLSL_TYPE_STRING(ctype) _Generic((ctype), \
  unsigned int: "uint",                           \
  float: "float")

#define APPEND_CONSTANT(name, cvar)     \
  ss << "const ";                       \
  ss << HLSL_TYPE_STRING(params->cvar); \
  ss << " " << name << " = ";           \
  ss << params->cvar << ";\n";

  APPEND_CONSTANT("NUM_THREADS_X", num_threads_x)
  APPEND_CONSTANT("NUM_THREADS_Y", num_threads_y)
  APPEND_CONSTANT("MAX_STACK_SIZE", max_stack_size)
  APPEND_CONSTANT("IMAGE_WIDTH", image_width)
  APPEND_CONSTANT("IMAGE_HEIGHT", image_height)
  APPEND_CONSTANT("SAMPLE_COUNT", spp)
  APPEND_CONSTANT("MAX_BOUNCES", max_bounces)
  APPEND_CONSTANT("CAMERA_ORIGIN_X", camera_position_x)
  APPEND_CONSTANT("CAMERA_ORIGIN_Y", camera_position_y)
  APPEND_CONSTANT("CAMERA_ORIGIN_Z", camera_position_z)
  APPEND_CONSTANT("CAMERA_FORWARD_X", camera_forward_x)
  APPEND_CONSTANT("CAMERA_FORWARD_Y", camera_forward_y)
  APPEND_CONSTANT("CAMERA_FORWARD_Z", camera_forward_z)
  APPEND_CONSTANT("CAMERA_UP_X", camera_up_x)
  APPEND_CONSTANT("CAMERA_UP_Y", camera_up_y)
  APPEND_CONSTANT("CAMERA_UP_Z", camera_up_z)
  APPEND_CONSTANT("CAMERA_VFOV", camera_vfov)
  APPEND_CONSTANT("RR_BOUNCE_OFFSET", rr_bounce_offset)
  APPEND_CONSTANT("RR_INV_MIN_TERM_PROB", rr_inv_min_term_prob)

  std::string fileName = "main.comp.hlsl";
  std::string filePath = s_shaderPath + "/" + fileName;
  *entryPoint = "CSMain";

  std::string source = ss.str() + "#include \"" + fileName + "\"\n";

  return s_shaderCompiler->compileHlslToSpv(source, filePath, *entryPoint, spvSize, spv);
}
