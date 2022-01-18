#include "IShaderCompiler.h"

namespace sg
{
  IShaderCompiler::IShaderCompiler(std::string_view shaderPath)
    : m_shaderPath(shaderPath)
  {
  }

  IShaderCompiler::~IShaderCompiler()
  {
  }
}
