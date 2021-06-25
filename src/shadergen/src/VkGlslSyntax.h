//
// TM & (c) 2017 Lucasfilm Entertainment Company Ltd. and Lucasfilm Ltd.
// All rights reserved. See LICENSE.txt for license.
//

#pragma once

#include <MaterialXGenShader/Syntax.h>

namespace mx = MaterialX;

namespace shadergen
{
  class VkGlslSyntax : public mx::Syntax
  {
  public:
    static const mx::string INPUT_QUALIFIER;
    static const mx::string OUTPUT_QUALIFIER;
    static const mx::string UNIFORM_QUALIFIER;
    static const mx::string CONSTANT_QUALIFIER;
    static const mx::string FLAT_QUALIFIER;
    static const mx::string SOURCE_FILE_EXTENSION;

    static const mx::StringVec VEC2_MEMBERS;
    static const mx::StringVec VEC3_MEMBERS;
    static const mx::StringVec VEC4_MEMBERS;

  public:
    explicit VkGlslSyntax();

    static mx::SyntaxPtr create() { return std::make_shared<VkGlslSyntax>(); }

  public:
    const mx::string& getInputQualifier() const override { return INPUT_QUALIFIER; }
    const mx::string& getOutputQualifier() const override { return OUTPUT_QUALIFIER; }
    const mx::string& getConstantQualifier() const override { return CONSTANT_QUALIFIER; };
    const mx::string& getUniformQualifier() const override { return UNIFORM_QUALIFIER; };
    const mx::string& getSourceFileExtension() const override { return SOURCE_FILE_EXTENSION; };

    bool typeSupported(const mx::TypeDesc* type) const override;

    /// Given an input specification attempt to remap this to an enumeration which is accepted by
    /// the shader generator. The enumeration may be converted to a different type than the input.
    bool remapEnumeration(const mx::string& value,
                          const mx::TypeDesc* type,
                          const mx::string& enumNames,
                          std::pair<const mx::TypeDesc*, mx::ValuePtr>& result) const override;
  };
}
