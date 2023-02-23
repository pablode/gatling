//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "MtlxDocumentPatcher.h"

#include <MaterialXCore/Types.h>

#include <unordered_set>
#include <assert.h>
#include <ctype.h>

namespace mx = MaterialX;

const char* TYPE_COLOR3 = "color3";
const char* TYPE_VECTOR3 = "vector3";

void _SanitizeFilePath(std::string& path)
{
  // The MDL SDK does not take raw OS paths. First, only forward-facing slashes are allowed.
  std::replace(path.begin(), path.end(), '\\', '/');

  // Second, only UNIX-style absolute paths ('/' prefix, no double colon) are valid.
  bool hasDriveSpecifier = path.size() >= 2 && path[1] == ':';

  if (hasDriveSpecifier)
  {
    path[1] = path[0];
    path[0] = '/';
  }
}

void _SanitizeFilePaths(mx::DocumentPtr document)
{
  for (auto treeIt = document->traverseTree(); treeIt != mx::TreeIterator::end(); ++treeIt)
  {
    mx::ElementPtr elem = treeIt.getElement();

    mx::PortElementPtr portElem = elem->asA<mx::PortElement>();
    if (!portElem)
    {
      continue;
    }

    std::string portType = portElem->getType();
    if (portType != mx::FILENAME_TYPE_STRING)
    {
      continue;
    }

    mx::ValuePtr valuePtr = portElem->getValue();
    if (!valuePtr)
    {
      continue;
    }

    std::string path = valuePtr->asA<std::string>();

    _SanitizeFilePath(path);

    portElem->setValue(path, mx::FILENAME_TYPE_STRING);
  }
}

mx::PortElementPtr _GetPortInterface(mx::PortElementPtr port)
{
  mx::ElementPtr parent = port->getParent();
  if (!parent)
  {
    return nullptr;
  }

  mx::NodePtr node = parent->asA<mx::Node>();
  if (!node)
  {
    return nullptr;
  }

  mx::NodeDefPtr nodeDef = node->getNodeDef(mx::EMPTY_STRING, true);
  if (!nodeDef)
  {
    return nullptr;
  }

  return nodeDef->getChildOfType<mx::PortElement>(port->getName());
}

std::string _GetPortType(mx::PortElementPtr port)
{
  mx::PortElementPtr portInterface = _GetPortInterface(port);

  return portInterface ? portInterface->getType() : port->getType();
}

// Workaround for an implicit type conversion issue between vector3 and color3 that occurs in code
// generated by the MDL backend: https://github.com/AcademySoftwareFoundation/MaterialX/issues/1038
void _PatchColor3Vector3Mismatch(mx::DocumentPtr document, mx::InputPtr input, mx::OutputPtr output)
{
  bool isInputColor3 = _GetPortType(input) == TYPE_COLOR3;

  bool isPatchable = (isInputColor3 && _GetPortType(output) == TYPE_VECTOR3) ||
                     (_GetPortType(input) == TYPE_VECTOR3 && _GetPortType(output) == TYPE_COLOR3);

  if (!isPatchable)
  {
    return;
  }

  std::string nodeCategory = "convert";
  std::string nodeName = mx::EMPTY_STRING; // auto-assign
  std::string nodeType = isInputColor3 ? TYPE_COLOR3 : TYPE_VECTOR3;
  mx::NodePtr node = document->addNode(nodeCategory, nodeName, nodeType);

  mx::InputPtr convertInput = node->addInput("in");
  convertInput->setConnectedOutput(output);

  // Can't clear because we need to preserve other attributes like 'colorspace'.
  input->removeAttribute(mx::PortElement::OUTPUT_ATTRIBUTE);
  input->removeAttribute(mx::PortElement::NODE_GRAPH_ATTRIBUTE);
  input->setType(nodeType);
  input->setConnectedNode(node);
}

void _PatchColor3Vector3Mismatches(mx::DocumentPtr document)
{
  for (auto treeIt = document->traverseTree(); treeIt != mx::TreeIterator::end(); ++treeIt)
  {
    mx::ElementPtr elem = treeIt.getElement();

    mx::InputPtr input = elem->asA<mx::Input>();
    if (!input)
    {
      continue;
    }

    mx::OutputPtr output = input->getConnectedOutput();
    if (!output)
    {
      continue;
    }

    _PatchColor3Vector3Mismatch(document, input, output);
  }
}

// HACK/FIXME:
// One big limitation of the MDL backend is currently that geompropvalue
// reader nodes are not implemented (they return a value of zero). By
// removing them, the default geomprop (e.g. UV0) is used, provided by the
// MDL state, which we can fill by anticipating certain geomprops/primvars
// on the Hydra side (e.g. the 'st' primvar). This way, we can still get
// proper texture coordinates in MOST cases, but not all.
void _PatchGeomprops(mx::DocumentPtr document)
{
  for (auto treeIt = document->traverseTree(); treeIt != mx::TreeIterator::end(); ++treeIt)
  {
    mx::ElementPtr elem = treeIt.getElement();

    mx::NodePtr node = elem->asA<mx::Node>();
    if (!node)
    {
      continue;
    }

    const mx::string& category = node->getCategory();

    if (category == "geompropvalue" || category == "UsdPrimvarReader")
    {
      document->removeNode(node->getName());
      continue;
    }

    if (category != "image" && category != "tiledimage")
    {
      continue;
    }

    auto texCoordInput = node->getActiveInput("texcoord");
    if (texCoordInput)
    {
      node->removeInput(texCoordInput->getName());
    }
  }
}

// According to the UsdPreviewSurface spec, the UsdUVTexture node has a sourceColorSpace input,
// which can take on the values 'raw', 'sRGB' and 'auto':
// https://graphics.pixar.com/usd/release/spec_usdpreviewsurface.html#texture-reader
//
// The MaterialX implementation does not provide this input, because color space transformations
// are supposed to be handled by node _attributes_ instead of inputs. Attributes can not be set
// dynamically. To work around the incompatibility of both approaches, this function replaces
// said input with the corresponding 'colorspace' attribute.
void _PatchUsdUVTextureSourceColorSpaces(mx::DocumentPtr document)
{
  for (auto treeIt = document->traverseTree(); treeIt != mx::TreeIterator::end(); ++treeIt)
  {
    mx::ElementPtr elem = treeIt.getElement();

    mx::InputPtr textureInput = elem->asA<mx::Input>();
    if (!textureInput || textureInput->hasColorSpace())
    {
      continue;
    }

    mx::ElementPtr upstreamElem = textureInput->getParent();
    mx::NodePtr downstreamNode = textureInput->getConnectedNode();
    if (!upstreamElem || !downstreamNode || downstreamNode->hasColorSpace())
    {
      continue;
    }

    mx::NodePtr upstreamNode = upstreamElem->asA<mx::Node>();
    if (!upstreamNode)
    {
      continue;
    }

    const mx::string& downstreamCategory = downstreamNode->getCategory();
    if (downstreamCategory != "UsdUVTexture")
    {
      continue;
    }

    mx::InputPtr colorSpaceInput = downstreamNode->getActiveInput("sourceColorSpace");
    mx::string textureInputName = textureInput->getName();

    std::string colorSpaceString = colorSpaceInput ? colorSpaceInput->getValueString() : "auto";

    const mx::string& upstreamCategory = upstreamNode->getCategory();
    bool isUpstreamUsdPreviewSurface = (upstreamCategory == "UsdPreviewSurface");

    bool isUsdPreviewSurfaceSrgbInput = isUpstreamUsdPreviewSurface &&
      (textureInputName == "diffuseColor" || textureInputName == "emissiveColor" || textureInputName == "specularColor");

    // Not spec-conform but should be more correct in most cases.
    bool isSrgbColorSpace = (colorSpaceString == "sRGB") || (colorSpaceString == "auto" && isUsdPreviewSurfaceSrgbInput);

    textureInput->setColorSpace(isSrgbColorSpace ? "srgb_texture" : "lin_rec709");

    // Prevent any other kind of processing.
    if (colorSpaceInput)
    {
      downstreamNode->removeInput(colorSpaceInput->getName());
    }
  }
}

// Currently, the HdMtlxCreateMtlxDocumentFromHdNetwork helper function commonly used by Hydra render
// delegates that support MaterialX does not copy color spaces: https://github.com/PixarAnimationStudios/USD/issues/1523
// We work around this issue by setting an <image> node's colorspace attribute to sRGB if the node type
// is color3. If it isn't (but rather float/vec2/vec3/vec4), we mark it as linear.
//
// FIXME: remove this patching step once the issue is resolved
void _PatchImageSrgbColorSpaces(mx::DocumentPtr document)
{
  for (auto treeIt = document->traverseTree(); treeIt != mx::TreeIterator::end(); ++treeIt)
  {
    mx::ElementPtr elem = treeIt.getElement();

    mx::NodePtr node = elem->asA<mx::Node>();
    if (!node || node->hasColorSpace()) // don't overwrite color space, f.i. from above sourceColorSpace patching
    {
      continue;
    }

    const mx::string& category = node->getCategory();

    if (category != "image" && category != "tiledimage")
    {
      continue;
    }

    const mx::string& valueType = node->getType();

    node->setColorSpace(valueType == TYPE_COLOR3 ? "srgb_texture" : "lin_rec709");
  }
}

// MDL spec 1.7.2 17th Jan 2022, section 5.6
std::unordered_set<std::string_view> s_reservedMDLIdentifiers = {
  "annotation", "auto", "bool", "bool2", "bool3", "bool4", "break", "bsdf",
  "bsdf_measurement", "case", "cast", "color", "const", "continue", "default",
  "do", "double", "double2", "double2x2", "double2x3", "double3", "double3x2",
  "double3x3", "double3x4", "double4", "double4x3", "double4x4", "double4x2",
  "double2x4", "edf", "else", "enum", "export", "false", "float", "float2",
  "float2x2", "float2x3", "float3", "float3x2", "float3x3", "float3x4",
  "float4", "float4x3", "float4x4", "float4x2", "float2x4", "for", "hair_bsdf",
  "if", "import", "in", "int", "int2", "int3", "int4", "intensity_mode",
  "intensity_power", "intensity_radiant_exitance", "let", "light_profile",
  "material", "material_emission", "material_geometry", "material_surface",
  "material_volume", "mdl", "module", "package", "return", "string", "struct",
  "switch", "texture_2d", "texture_3d", "texture_cube", "texture_ptex", "true",
  "typedef", "uniform", "using", "varying", "vdf", "while", "catch", "char",
  "class", "const_cast", "delete", "dynamic_cast", "explicit", "extern",
  "external", "foreach", "friend", "goto", "graph", "half", "half2", "half2x2",
  "half2x3", "half3", "half3x2", "half3x3", "half3x4", "half4", "half4x3",
  "half4x4", "half4x2", "half2x4", "inline", "inout", "lambda", "long",
  "mutable", "namespace", "native", "new", "operator", "out", "phenomenon",
  "private", "protected", "public", "reinterpret_cast", "sampler", "shader",
  "short", "signed", "sizeof", "static", "static_cast", "technique", "template",
  "this", "throw", "try", "typeid", "typename", "union", "unsigned", "virtual",
  "void", "volatile", "wchar_t"
};

// MDL spec section 5.5 and 5.6: "An identifier is an alphabetic character followed
// by a possibly empty sequence of alphabetic characters, decimal digits, and underscores,
// that is neither a typename nor a reserved word."
// https://raytracing-docs.nvidia.com/mdl/specification/MDL_spec_1.7.2_17Jan2022.pdf
bool _MakeValidMDLIdentifier(std::string& str)
{
  assert(!str.empty());
  bool strChanged = false;

  // Replace all chars that are not 1) alphabetic or 2) decimal or 3) underscores with underscores
  for (size_t i = 0; i < str.size(); i++)
  {
    char c = str[i];

    if (isalnum(c) || c == '_')
    {
      continue;
    }

    str[i] = '_';
    strChanged |= true;
  }

  bool invalidFirstChar = !isalpha(str[0]);
  bool isReservedKeyword = s_reservedMDLIdentifiers.find(str) != s_reservedMDLIdentifiers.end();

  bool usePrefix = invalidFirstChar || isReservedKeyword;
  if (usePrefix)
  {
    str = std::string("GAT" + str);
    strChanged |= true;
  }

  return strChanged;
}

void _PatchNodeNames(mx::DocumentPtr document)
{
  for (auto treeIt = document->traverseTree(); treeIt != mx::TreeIterator::end(); ++treeIt)
  {
    mx::ElementPtr elem = treeIt.getElement();

    mx::NodePtr node = elem->asA<mx::Node>();
    if (!node)
    {
      continue;
    }

    std::string newName = node->getName();

    if (!_MakeValidMDLIdentifier(newName))
    {
      continue;
    }

    std::string oldName = node->getName();

    // FIXME: this 'node renaming' algorithm works, but is not likely to cover all cases.
    // Ideally, there should a MaterialX library function for this purpose.
    for (mx::NodeGraphPtr nodeGraph : document->getNodeGraphs())
    {
      for (mx::OutputPtr output : nodeGraph->getOutputs())
      {
        if (output->getConnectedNode() == node)
        {
          output->setNodeName(newName);
        }
      }
    }
    for (mx::OutputPtr output : node->getOutputs())
    {
      if (output->getNodeName() == oldName)
      {
        output->setNodeName(newName);
      }
    }

    node->setName(newName);
  }
}

namespace gi::sg
{
  void MtlxDocumentPatcher::patch(MaterialX::DocumentPtr document)
  {
    _SanitizeFilePaths(document);

    _PatchColor3Vector3Mismatches(document);

    _PatchUsdUVTextureSourceColorSpaces(document);

    _PatchGeomprops(document);

    _PatchImageSrgbColorSpaces(document);

    _PatchNodeNames(document);
  }
}
