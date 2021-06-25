//
// TM & (c) 2017 Lucasfilm Entertainment Company Ltd. and Lucasfilm Ltd.
// All rights reserved. See LICENSE.txt for license.
//

#include "VkGlslSyntax.h"

#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Nodes/ThinFilmNode.h>

namespace mx = MaterialX;

namespace shadergen
{
  namespace
  {
    // Since GLSL doesn't support strings we use integers instead.
    // TODO: Support options strings by converting to a corresponding enum integer
    class GlslStringTypeSyntax : public mx::StringTypeSyntax
    {
    public:
      GlslStringTypeSyntax() : StringTypeSyntax("int", "0", "0") {}

      mx::string getValue(const mx::Value& /*value*/, bool /*uniform*/) const override
      {
        return "0";
      }
    };

    class GlslArrayTypeSyntax : public mx::ScalarTypeSyntax
    {
    public:
      GlslArrayTypeSyntax(const mx::string& name) :
        mx::ScalarTypeSyntax(name, mx::EMPTY_STRING, mx::EMPTY_STRING, mx::EMPTY_STRING)
      {}

      mx::string getValue(const mx::Value& value, bool /*uniform*/) const override
      {
        size_t arraySize = getSize(value);
        if (arraySize > 0)
        {
          return _name + "[" + std::to_string(arraySize) + "](" + value.getValueString() + ")";
        }
        return mx::EMPTY_STRING;
      }

      mx::string getValue(const mx::StringVec& values, bool /*uniform*/) const override
      {
        if (values.empty())
        {
          throw mx::ExceptionShaderGenError("No values given to construct an array value");
        }

        mx::string result = _name + "[" + std::to_string(values.size()) + "](" + values[0];
        for (size_t i = 1; i < values.size(); ++i)
        {
          result += ", " + values[i];
        }
        result += ")";

        return result;
      }

    protected:
      virtual size_t getSize(const mx::Value& value) const = 0;
    };

    class GlslFloatArrayTypeSyntax : public GlslArrayTypeSyntax
    {
    public:
      explicit GlslFloatArrayTypeSyntax(const mx::string& name) :
        GlslArrayTypeSyntax(name)
      {}

    protected:
      size_t getSize(const mx::Value& value) const override
      {
        mx::vector<float> valueArray = value.asA<mx::vector<float>>();
        return valueArray.size();
      }
    };

    class GlslIntegerArrayTypeSyntax : public GlslArrayTypeSyntax
    {
    public:
      explicit GlslIntegerArrayTypeSyntax(const mx::string& name) :
        GlslArrayTypeSyntax(name)
      {}

    protected:
      size_t getSize(const mx::Value& value) const override
      {
        mx::vector<int> valueArray = value.asA<mx::vector<int>>();
        return valueArray.size();
      }
    };
  }

  const mx::string VkGlslSyntax::INPUT_QUALIFIER = "in";
  const mx::string VkGlslSyntax::OUTPUT_QUALIFIER = "out";
  const mx::string VkGlslSyntax::UNIFORM_QUALIFIER = "uniform";
  const mx::string VkGlslSyntax::CONSTANT_QUALIFIER = "const";
  const mx::string VkGlslSyntax::FLAT_QUALIFIER = "flat";
  const mx::string VkGlslSyntax::SOURCE_FILE_EXTENSION = ".glsl";
  const mx::StringVec VkGlslSyntax::VEC2_MEMBERS = { ".x", ".y" };
  const mx::StringVec VkGlslSyntax::VEC3_MEMBERS = { ".x", ".y", ".z" };
  const mx::StringVec VkGlslSyntax::VEC4_MEMBERS = { ".x", ".y", ".z", ".w" };

  VkGlslSyntax::VkGlslSyntax()
  {
    // Add in all reserved words and keywords in GLSL
    registerReservedWords(
      {
          "centroid", "flat", "smooth", "noperspective", "patch", "sample",
          "break", "continue", "do", "for", "while", "switch", "case", "default",
          "if", "else,", "subroutine", "in", "out", "inout",
          "float", "double", "int", "void", "bool", "true", "false",
          "invariant", "discard", "return",
          "mat2", "mat3", "mat4", "dmat2", "dmat3", "dmat4",
          "mat2x2", "mat2x3", "mat2x4", "dmat2x2", "dmat2x3", "dmat2x4",
          "mat3x2", "mat3x3", "mat3x4", "dmat3x2", "dmat3x3", "dmat3x4",
          "mat4x2", "mat4x3", "mat4x4", "dmat4x2", "dmat4x3", "dmat4x4",
          "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3", "bvec4", "dvec2", "dvec3", "dvec4",
          "uint", "uvec2", "uvec3", "uvec4",
          "lowp", "mediump", "highp", "precision",
          "sampler1D", "sampler2D", "sampler3D", "samplerCube",
          "sampler1DShadow", "sampler2DShadow", "samplerCubeShadow",
          "sampler1DArray", "sampler2DArray",
          "sampler1DArrayShadow", "sampler2DArrayShadow",
          "isampler1D", "isampler2D", "isampler3D", "isamplerCube",
          "isampler1DArray", "isampler2DArray",
          "usampler1D", "usampler2D", "usampler3D", "usamplerCube",
          "usampler1DArray", "usampler2DArray",
          "sampler2DRect", "sampler2DRectShadow", "isampler2DRect", "usampler2DRect",
          "samplerBuffer", "isamplerBuffer", "usamplerBuffer",
          "sampler2DMS", "isampler2DMS", "usampler2DMS",
          "sampler2DMSArray", "isampler2DMSArray", "usampler2DMSArray",
          "samplerCubeArray", "samplerCubeArrayShadow", "isamplerCubeArray", "usamplerCubeArray",
          "common", "partition", "active", "asm",
          "struct", "class", "union", "enum", "typedef", "template", "this", "packed", "goto",
          "inline", "noinline", "volatile", "public", "static", "extern", "external", "interface",
          "long", "short", "half", "fixed", "unsigned", "superp", "input", "output",
          "hvec2", "hvec3", "hvec4", "fvec2", "fvec3", "fvec4",
          "sampler3DRect", "filter",
          "image1D", "image2D", "image3D", "imageCube",
          "iimage1D", "iimage2D", "iimage3D", "iimageCube",
          "uimage1D", "uimage2D", "uimage3D", "uimageCube",
          "image1DArray", "image2DArray",
          "iimage1DArray", "iimage2DArray", "uimage1DArray", "uimage2DArray",
          "image1DShadow", "image2DShadow",
          "image1DArrayShadow", "image2DArrayShadow",
          "imageBuffer", "iimageBuffer", "uimageBuffer",
          "sizeof", "cast", "namespace", "using", "row_major"
      }
    );

    // Register restricted tokens in GLSL
    mx::StringMap tokens;
    tokens["__"] = "_";
    tokens["gl_"] = "gll";
    tokens["webgl_"] = "webgll";
    tokens["_webgl"] = "wwebgl";
    registerInvalidTokens(tokens);

    //
    // Register syntax handlers for each data type.
    //

    registerTypeSyntax
    (
      mx::Type::FLOAT,
      std::make_shared<mx::ScalarTypeSyntax>(
        "float",
        "0.0",
        "0.0")
    );

    registerTypeSyntax
    (
      mx::Type::FLOATARRAY,
      std::make_shared<GlslFloatArrayTypeSyntax>(
        "float")
    );

    registerTypeSyntax
    (
      mx::Type::INTEGER,
      std::make_shared<mx::ScalarTypeSyntax>(
        "int",
        "0",
        "0")
    );

    registerTypeSyntax
    (
      mx::Type::INTEGERARRAY,
      std::make_shared<GlslIntegerArrayTypeSyntax>(
        "int")
    );

    registerTypeSyntax
    (
      mx::Type::BOOLEAN,
      std::make_shared<mx::ScalarTypeSyntax>(
        "bool",
        "false",
        "false")
    );

    registerTypeSyntax
    (
      mx::Type::COLOR3,
      std::make_shared<mx::AggregateTypeSyntax>(
        "vec3",
        "vec3(0.0)",
        "vec3(0.0)",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        VEC3_MEMBERS)
    );

    registerTypeSyntax
    (
      mx::Type::COLOR4,
      std::make_shared<mx::AggregateTypeSyntax>(
        "vec4",
        "vec4(0.0)",
        "vec4(0.0)",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        VEC4_MEMBERS)
    );

    registerTypeSyntax
    (
      mx::Type::VECTOR2,
      std::make_shared<mx::AggregateTypeSyntax>(
        "vec2",
        "vec2(0.0)",
        "vec2(0.0)",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        VEC2_MEMBERS)
    );

    registerTypeSyntax
    (
      mx::Type::VECTOR3,
      std::make_shared<mx::AggregateTypeSyntax>(
        "vec3",
        "vec3(0.0)",
        "vec3(0.0)",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        VEC3_MEMBERS)
    );

    registerTypeSyntax
    (
      mx::Type::VECTOR4,
      std::make_shared<mx::AggregateTypeSyntax>(
        "vec4",
        "vec4(0.0)",
        "vec4(0.0)",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        VEC4_MEMBERS)
    );

    registerTypeSyntax
    (
      mx::Type::MATRIX33,
      std::make_shared<mx::AggregateTypeSyntax>(
        "mat3",
        "mat3(1.0)",
        "mat3(1.0)")
    );

    registerTypeSyntax
    (
      mx::Type::MATRIX44,
      std::make_shared<mx::AggregateTypeSyntax>(
        "mat4",
        "mat4(1.0)",
        "mat4(1.0)")
    );

    registerTypeSyntax
    (
      mx::Type::STRING,
      std::make_shared<GlslStringTypeSyntax>()
    );

    registerTypeSyntax
    (
      mx::Type::FILENAME,
      std::make_shared<mx::ScalarTypeSyntax>(
        "sampler2D",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING)
    );

    registerTypeSyntax
    (
      mx::Type::BSDF,
      std::make_shared<mx::AggregateTypeSyntax>(
        "BSDF",
        "BSDF(0.0)",
        "BSDF(0.0)",
        "vec3")
    );

    registerTypeSyntax
    (
      mx::Type::EDF,
      std::make_shared<mx::AggregateTypeSyntax>(
        "EDF",
        "EDF(0.0)",
        "EDF(0.0)",
        "vec3")
    );

    registerTypeSyntax
    (
      mx::Type::VDF,
      std::make_shared<mx::AggregateTypeSyntax>(
        "VDF",
        "VDF(vec3(0.0),vec3(0.0))",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        "struct VDF { vec3 absorption; vec3 scattering; };")
    );

    registerTypeSyntax
    (
      mx::Type::SURFACESHADER,
      std::make_shared<mx::AggregateTypeSyntax>(
        "surfaceshader",
        "surfaceshader(vec3(0.0),vec3(0.0))",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        "struct surfaceshader { vec3 color; vec3 transparency; };")
    );

    registerTypeSyntax
    (
      mx::Type::VOLUMESHADER,
      std::make_shared<mx::AggregateTypeSyntax>(
        "volumeshader",
        "volumeshader(VDF(vec3(0.0),vec3(0.0)),EDF(0.0))",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        "struct volumeshader { VDF vdf; EDF edf; };")
    );

    registerTypeSyntax
    (
      mx::Type::DISPLACEMENTSHADER,
      std::make_shared<mx::AggregateTypeSyntax>(
        "displacementshader",
        "displacementshader(vec3(0.0),1.0)",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        "struct displacementshader { vec3 offset; float scale; };")
    );

    registerTypeSyntax
    (
      mx::Type::LIGHTSHADER,
      std::make_shared<mx::AggregateTypeSyntax>(
        "lightshader",
        "lightshader(vec3(0.0),vec3(0.0))",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        "struct lightshader { vec3 intensity; vec3 direction; };")
    );

    registerTypeSyntax
    (
      mx::Type::THINFILM,
      std::make_shared<mx::AggregateTypeSyntax>(
        "thinfilm",
        "thinfilm(0.0,1.5)",
        mx::EMPTY_STRING,
        mx::EMPTY_STRING,
        "struct thinfilm { float thickness; float ior; };")
    );
  }

  bool VkGlslSyntax::typeSupported(const mx::TypeDesc* type) const
  {
    return type != mx::Type::STRING;
  }

  bool VkGlslSyntax::remapEnumeration(const mx::string& value, const mx::TypeDesc* type, const mx::string& enumNames, std::pair<const mx::TypeDesc*, mx::ValuePtr>& result) const
  {
    // Early out if not an enum input.
    if (enumNames.empty())
    {
      return false;
    }

    // Don't convert already supported types
    // or filenames and arrays.
    if (typeSupported(type) ||
      type == mx::Type::FILENAME || (type && type->isArray()))
    {
      return false;
    }

    // For GLSL we always convert to integer,
    // with the integer value being an index into the enumeration.
    result.first = mx::Type::INTEGER;
    result.second = nullptr;

    // Try remapping to an enum value.
    if (!value.empty())
    {
      mx::StringVec valueElemEnumsVec = mx::splitString(enumNames, ",");
      for (size_t i = 0; i < valueElemEnumsVec.size(); i++)
      {
        valueElemEnumsVec[i] = mx::trimSpaces(valueElemEnumsVec[i]);
      }
      auto pos = std::find(valueElemEnumsVec.begin(), valueElemEnumsVec.end(), value);
      if (pos == valueElemEnumsVec.end())
      {
        throw mx::ExceptionShaderGenError("Given value '" + value + "' is not a valid enum value for input.");
      }
      const int index = static_cast<int>(std::distance(valueElemEnumsVec.begin(), pos));
      result.second = mx::Value::createValue<int>(index);
    }

    return true;
  }
}
