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

#include "materialNetworkCompiler.h"
#include "previewSurfaceNetworkPatcher.h"
#include "tokens.h"

#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usd/sdr/shaderProperty.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXCore/Material.h>
#include <MaterialXCore/Definition.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <gtl/gb/Fmt.h>
#include <gtl/gi/Gi.h>

namespace mx = MaterialX;

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  // USD tokens
  (UsdPreviewSurface)
  (UsdPrimvarReader_float)
  (UsdPrimvarReader_float2)
  (UsdPrimvarReader_float3)
  (UsdPrimvarReader_float4)
  (UsdPrimvarReader_int)
  (UsdPrimvarReader_string)
  (UsdPrimvarReader_normal)
  (UsdPrimvarReader_point)
  (UsdPrimvarReader_vector)
  (UsdPrimvarReader_matrix)
  (UsdTransform2d)
  (UsdUVTexture)
  (normal)
  (wrapS)
  (wrapT)
  (black)
  (clamp)
  (repeat)
  (mirror)
  (sourceColorSpace)
  (raw)
  (rgb)
  (sRGB)
  (in)
  (result)
  (out)
  ((_auto, "auto"))
  // MaterialX tokens
  (ND_UsdPreviewSurface_surfaceshader)
  (ND_UsdPrimvarReader_integer)
  (ND_UsdPrimvarReader_boolean)
  (ND_UsdPrimvarReader_string)
  (ND_UsdPrimvarReader_float)
  (ND_UsdPrimvarReader_vector2)
  (ND_UsdPrimvarReader_vector3)
  (ND_UsdPrimvarReader_vector4)
  (ND_UsdPrimvarReader_matrix44)
  (ND_UsdTransform2d)
  (ND_UsdUVTexture)
  (ND_convert_color3_vector3)
  (ND_convert_vector3_color3)
  (periodic)
  (srgb_texture)
  (lin_rec709)
);

static std::unordered_map<TfToken, TfToken, TfToken::HashFunctor> _usdMtlxNodeTypeIdMappings = {
  { _tokens->UsdPreviewSurface,       _tokens->ND_UsdPreviewSurface_surfaceshader },
  { _tokens->UsdUVTexture,            _tokens->ND_UsdUVTexture                    },
  { _tokens->UsdTransform2d,          _tokens->ND_UsdTransform2d                  },
  { _tokens->UsdPrimvarReader_float,  _tokens->ND_UsdPrimvarReader_float          },
  { _tokens->UsdPrimvarReader_float2, _tokens->ND_UsdPrimvarReader_vector2        },
  { _tokens->UsdPrimvarReader_float3, _tokens->ND_UsdPrimvarReader_vector3        },
  { _tokens->UsdPrimvarReader_float4, _tokens->ND_UsdPrimvarReader_vector4        },
  { _tokens->UsdPrimvarReader_int,    _tokens->ND_UsdPrimvarReader_integer        },
  { _tokens->UsdPrimvarReader_string, _tokens->ND_UsdPrimvarReader_string         },
  { _tokens->UsdPrimvarReader_normal, _tokens->ND_UsdPrimvarReader_vector3        },
  { _tokens->UsdPrimvarReader_point,  _tokens->ND_UsdPrimvarReader_vector3        },
  { _tokens->UsdPrimvarReader_vector, _tokens->ND_UsdPrimvarReader_vector3        },
  { _tokens->UsdPrimvarReader_matrix, _tokens->ND_UsdPrimvarReader_matrix44       }
};

// MaterialX shading network errors are not always fatal -- and they may not be even propagated to the user.
// This leads to networks being authored that work and look fine in one renderer (usually OpenGL-based) but
// may not in another renderer (an anti-goal of MaterialX!).
//
// This function patches the specific case of node connections being authored between inputs of type color3f
// and outputs of type vector3f (and the other way around). These types of connections work fine in OpenGL-based
// renderers because colors are aliased as vec3s in GLSL.
// But this is not the case with MDL - colors are distinct types and implicit conversion is illegal.
// As a result, the MDL SDK throws compilation errors and we can't render the material.
// The issue has been reported here: https://github.com/AcademySoftwareFoundation/MaterialX/issues/1038
//
// The problem occurs mainly when translating normal-mapped UsdPreviewSurface nodes to their MaterialX equivalents
// (a common scenario). The 'normal' input of the UsdPreviewSurface node expects a vector3, while the connected
// UsdUVTexture node output is of type color3.
// But also external MaterialX-HdShade networks may exhibit this behaviour. An in-the-wild example is the Karma
// tutorial 'A Beautiful Game': https://www.sidefx.com/tutorials/karma-a-beautiful-game/ rendered within Houdini.
//
// We implement this patch on the MaterialX document level too, however we replicate it here so that HdMtlx does not
// throw validation errors due to mismatching NodeDefs.
void _PatchMaterialXColor3Vector3Mismatches(HdMaterialNetwork2& network)
{
  SdrRegistry& sdrRegistry = SdrRegistry::GetInstance();

  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;

    SdrShaderNodeConstPtr sdrNode = sdrRegistry.GetShaderNodeByIdentifierAndType(node.nodeTypeId, HdGatlingDiscoveryTypes->mtlx);
    if (!sdrNode)
    {
      continue;
    }

    auto& inputs = node.inputConnections;

    for (auto& input : inputs)
    {
      SdrShaderPropertyConstPtr sdrInput = sdrNode->GetShaderInput(input.first);

      auto ndrSdfType = sdrInput->GetTypeAsSdfType();
#if PXR_VERSION > 2408
      SdfValueTypeName inputType = ndrSdfType.GetSdfType();
#else
      SdfValueTypeName inputType = ndrSdfType.first;
#endif
      if (inputType == SdfValueTypeNames->Token)
      {
        continue;
      }

      bool isInputColor3 = (inputType == SdfValueTypeNames->Color3f);
      bool isInputFloat3 = (inputType == SdfValueTypeNames->Float3) ||
                           (inputType == SdfValueTypeNames->Vector3f) ||
                           (inputType == SdfValueTypeNames->Normal3f);

      for (HdMaterialConnection2& connection : input.second)
      {
        SdfPath upstreamNodePath = connection.upstreamNode;
        const HdMaterialNode2& upstreamNode = network.nodes[upstreamNodePath];

        SdrShaderNodeConstPtr upstreamSdrNode = sdrRegistry.GetShaderNodeByIdentifierAndType(upstreamNode.nodeTypeId, HdGatlingDiscoveryTypes->mtlx);
        if (!upstreamSdrNode)
        {
          continue;
        }

        SdrShaderPropertyConstPtr upstreamSdrOutput = upstreamSdrNode->GetShaderOutput(connection.upstreamOutputName);
        if (!upstreamSdrOutput)
        {
          continue;
        }

        auto upstreamNdrSdfType = upstreamSdrOutput->GetTypeAsSdfType();
#if PXR_VERSION > 2408
        SdfValueTypeName upstreamOutputType = upstreamNdrSdfType.GetSdfType();
#else
        SdfValueTypeName upstreamOutputType = upstreamNdrSdfType.first;
#endif
        if (upstreamOutputType == SdfValueTypeNames->Token)
        {
          continue;
        }

        bool isUpstreamColor3 = (upstreamOutputType == SdfValueTypeNames->Color3f);
        bool isUpstreamFloat3 = (upstreamOutputType == SdfValueTypeNames->Float3) ||
                                (upstreamOutputType == SdfValueTypeNames->Vector3f) ||
                                (upstreamOutputType == SdfValueTypeNames->Normal3f);

        bool mismatchCase1 = (isInputColor3 && isUpstreamFloat3);
        bool mismatchCase2 = (isInputFloat3 && isUpstreamColor3);

        if (!mismatchCase1 && !mismatchCase2)
        {
          continue;
        }

        SdfPath convertNodePath = upstreamNodePath;
        for (int i = 0; network.nodes.count(convertNodePath) > 0; i++)
        {
          std::string convertNodeName = GB_FMT("convert{}", i);
          convertNodePath = upstreamNodePath.AppendElementString(convertNodeName);
        }

        HdMaterialNode2 convertNode;
        convertNode.nodeTypeId = mismatchCase1 ? _tokens->ND_convert_vector3_color3 : _tokens->ND_convert_color3_vector3;
        convertNode.inputConnections[_tokens->in] = { { upstreamNodePath, connection.upstreamOutputName } };
        network.nodes[convertNodePath] = convertNode;

        connection.upstreamNode = convertNodePath;
        connection.upstreamOutputName = _tokens->out;
      }
    }
  }
}

bool _ConvertUsdNodesToMtlxNodes(HdMaterialNetwork2& network)
{
  // First pass: substitute UsdUVTexture:sourceColorSpace input with parent input colorSpace attribute
  for (auto nodeIt = network.nodes.begin(); nodeIt != network.nodes.end(); nodeIt++)
  {
    TfToken& nodeTypeId = nodeIt->second.nodeTypeId;

    if (nodeTypeId != _tokens->UsdPreviewSurface)
    {
      continue;
    }

    auto handleUsdUVTextureSourceColorSpaceInput = [&](HdMaterialNode2& parentNode, TfToken input, HdMaterialNode2& node)
    {
      auto& parameters = node.parameters;

      auto sourceColorSpace = parameters.find(_tokens->sourceColorSpace);
      if (sourceColorSpace == parameters.end())
      {
        return;
      }

      TfToken colorSpaceInputName(SdfPath::JoinIdentifier(SdfFieldKeys->ColorSpace, input));

      if (sourceColorSpace->second == _tokens->raw)
      {
        parentNode.parameters[colorSpaceInputName] = _tokens->lin_rec709;
      }
      else if (sourceColorSpace->second == _tokens->sRGB)
      {
        parentNode.parameters[colorSpaceInputName] = _tokens->srgb_texture;
      }
      else if (sourceColorSpace->second == _tokens->_auto)
      {
        // don't set color space explicitly
      }
      else
      {
        TF_CODING_ERROR("unsupported UsdUVTexture color space");
      }

      parameters.erase(sourceColorSpace);
    };

    for (const auto& inputConnections : nodeIt->second.inputConnections)
    {
      TfToken input = inputConnections.first;

      for (const HdMaterialConnection2& connection : inputConnections.second)
      {
        auto upstreamNode = network.nodes.find(connection.upstreamNode);

        if (upstreamNode == network.nodes.end())
        {
          continue;
        }

        if (upstreamNode->second.nodeTypeId != _tokens->UsdUVTexture)
        {
          continue;
        }

        handleUsdUVTextureSourceColorSpaceInput(nodeIt->second, input, upstreamNode->second);
      }
    }
  }

  // Second pass: patch certain outputs from 'result' (spec) to 'out' (MaterialX convention)
  for (auto nodeIt = network.nodes.begin(); nodeIt != network.nodes.end(); nodeIt++)
  {
    for (auto& inputConnection : nodeIt->second.inputConnections)
    {
      for (auto& connection : inputConnection.second)
      {
        auto upstreamNode = network.nodes.find(connection.upstreamNode);
        if (upstreamNode == network.nodes.end())
        {
          continue;
        }

        TfToken upstreamNodeTypeId = upstreamNode->second.nodeTypeId;
        if (upstreamNodeTypeId != _tokens->UsdPrimvarReader_float &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_float2 &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_float3 &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_float4 &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_int &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_string &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_normal &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_point &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_vector &&
            upstreamNodeTypeId != _tokens->UsdPrimvarReader_matrix &&
            upstreamNodeTypeId != _tokens->UsdTransform2d)
        {
          continue;
        }

        TfToken& upstreamOutputName = connection.upstreamOutputName;
        if (upstreamOutputName == _tokens->result)
        {
          upstreamOutputName = _tokens->out;
        }
      }
    }
  }

  // Third pass: substitute node names and parameters
  for (auto nodeIt = network.nodes.begin(); nodeIt != network.nodes.end(); nodeIt++)
  {
    TfToken& nodeTypeId = nodeIt->second.nodeTypeId;

    SdrRegistry& sdrRegistry = SdrRegistry::GetInstance();
    if (sdrRegistry.GetShaderNodeByIdentifierAndType(nodeTypeId, HdGatlingDiscoveryTypes->mtlx))
    {
      continue;
    }

    auto mappingIt = _usdMtlxNodeTypeIdMappings.find(nodeTypeId);
    if (mappingIt == _usdMtlxNodeTypeIdMappings.end())
    {
      TF_WARN("Unable to translate material node of type %s to MaterialX counterpart", nodeTypeId.GetText());
      return false;
    }

    if (nodeTypeId == _tokens->UsdUVTexture)
    {
      // MaterialX node inputs do not match the USD spec; we need to remap.
      auto convertWrapType = [](VtValue& wrapType)
      {
        auto wrapToken = wrapType.UncheckedGet<TfToken>();

        if (wrapToken == _tokens->black)
        {
          // It's internally mapped to 'constant' which uses the fallback color
          TF_WARN("UsdUVTexture wrap mode black is not fully supported");
        }
        else if (wrapToken == _tokens->mirror ||
                 wrapToken == _tokens->clamp)
        {
          // These are valid, do nothing.
        }
        else if (wrapToken == _tokens->repeat)
        {
          wrapType = _tokens->periodic;
        }
        else
        {
          TF_WARN("UsdUVTexture node has unsupported wrap mode %s", wrapToken.GetText());
          wrapType = _tokens->periodic;
        }
      };

      auto& parameters = nodeIt->second.parameters;
      auto wrapS = parameters.find(_tokens->wrapS);
      auto wrapT = parameters.find(_tokens->wrapT);

      if (wrapS != parameters.end())
      {
        convertWrapType(wrapS->second);
      }
      if (wrapT != parameters.end())
      {
        convertWrapType(wrapT->second);
      }
    }

    nodeTypeId = mappingIt->second;
  }

  return true;
}

bool _GetMaterialNetworkSurfaceTerminal(const HdMaterialNetwork2& network2, HdMaterialNode2& terminalNode, SdfPath& terminalPath)
{
  const auto& connectionIt = network2.terminals.find(HdMaterialTerminalTokens->surface);

  if (connectionIt == network2.terminals.end())
  {
    return false;
  }

  const HdMaterialConnection2& connection = connectionIt->second;

  terminalPath = connection.upstreamNode;

  const auto& nodeIt = network2.nodes.find(terminalPath);

  if (nodeIt == network2.nodes.end())
  {
    return false;
  }

  terminalNode = nodeIt->second;

  return true;
}

MaterialNetworkCompiler::MaterialNetworkCompiler(const mx::DocumentPtr mtlxStdLib)
  : _mtlxStdLib(mtlxStdLib)
{
}

GiMaterial* MaterialNetworkCompiler::CompileNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const
{
  GiMaterial* result = _TryCompileMdlNetwork(id, network);

  if (!result)
  {
    HdMaterialNetwork2 patchedNetwork = network;

    PreviewSurfaceNetworkPatcher patcher;
    patcher.Patch(patchedNetwork);

    result = _TryCompileMtlxNetwork(id, patchedNetwork);
  }

  return result;
}

GiMaterial* MaterialNetworkCompiler::_TryCompileMdlNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const
{
  if (network.nodes.size() != 1)
  {
    return nullptr;
  }

  const HdMaterialNode2& node = network.nodes.begin()->second;

  SdrRegistry& sdrRegistry = SdrRegistry::GetInstance();
  SdrShaderNodeConstPtr sdrNode = sdrRegistry.GetShaderNodeByIdentifier(node.nodeTypeId);

  if (!sdrNode || sdrNode->GetContext() != HdGatlingNodeContexts->mdl)
  {
    return nullptr;
  }

  const NdrTokenMap& metadata = sdrNode->GetMetadata();
  const auto& subIdentifierIt = metadata.find(HdGatlingNodeMetadata->subIdentifier);
  TF_AXIOM(subIdentifierIt != metadata.end());

  const std::string& subIdentifier = (*subIdentifierIt).second;
  const std::string& fileUri = sdrNode->GetResolvedImplementationURI();

  return giCreateMaterialFromMdlFile(id.GetText(), fileUri.c_str(), subIdentifier.c_str());
}

GiMaterial* MaterialNetworkCompiler::_TryCompileMtlxNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const
{
  HdMaterialNetwork2 mtlxNetwork = network;
  if (!_ConvertUsdNodesToMtlxNodes(mtlxNetwork))
  {
    return nullptr;
  }

  _PatchMaterialXColor3Vector3Mismatches(mtlxNetwork);

  mx::DocumentPtr doc = _CreateMaterialXDocumentFromNetwork(id, mtlxNetwork);
  if (!doc)
  {
    return nullptr;
  }

  return giCreateMaterialFromMtlxDoc(id.GetText(), doc);
}

mx::DocumentPtr MaterialNetworkCompiler::_CreateMaterialXDocumentFromNetwork(const SdfPath& id,
                                                                             const HdMaterialNetwork2& network) const
{
  HdMaterialNode2 terminalNode;
  SdfPath terminalPath;
  if (!_GetMaterialNetworkSurfaceTerminal(network, terminalNode, terminalPath))
  {
    TF_WARN("Unable to find surface terminal for material network");
    return nullptr;
  }

#if PXR_VERSION >= 2211
  HdMtlxTexturePrimvarData mxHdData;
#else
  std::set<SdfPath> hdTextureNodes;
  mx::StringMap mxHdTextureMap;
#endif

  return HdMtlxCreateMtlxDocumentFromHdNetwork(
    network,
    terminalNode,
    terminalPath,
    id,
    _mtlxStdLib,
#if PXR_VERSION >= 2211
    &mxHdData
#else
    &hdTextureNodes,
    &mxHdTextureMap
#endif
  );
}

PXR_NAMESPACE_CLOSE_SCOPE
