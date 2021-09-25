#include "IShaderCompiler.h"

namespace sg
{
  IShaderCompiler::IShaderCompiler(const std::string& shaderPath)
    : m_shaderPath(shaderPath)
  {
  }

  IShaderCompiler::~IShaderCompiler()
  {
  }
}
