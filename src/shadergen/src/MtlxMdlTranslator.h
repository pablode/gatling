#pragma once

#include <stdint.h>
#include <string>
#include <memory>

#include <MaterialXCore/Document.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>

namespace sg
{
  class MtlxMdlTranslator
  {
  public:
    explicit MtlxMdlTranslator(const char* mtlxlibPath);

  public:
    bool translate(const char* mtlxSrc, std::string& mdlSrc, std::string& subIdentifier);

  private:
    MaterialX::DocumentPtr m_stdLib;
    MaterialX::ShaderGeneratorPtr m_shaderGen;
    std::unique_ptr<MaterialX::GenContext> m_context;
  };
}
