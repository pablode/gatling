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

#include "MaterialNetworkCompiler.h"

#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXCore/Material.h>
#include <MaterialXCore/Definition.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <gi.h>

#include "PreviewSurfaceNetworkPatcher.h"
#include "Tokens.h"

namespace mx = MaterialX;

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  // USD node type ids
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
  (out)
  ((_auto, "auto"))
  // MaterialX USD node type equivalents
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

// Unfortunately the UsdPreviewSurface standard nodes can't be mapped to MaterialX UsdPreviewSurface
// implementation nodes as-is. This is because the 'normal' input of the UsdPreviewSurface node expects
// a vector3, while UsdUVTexture nodes only output color3 -- which can't be implicitly converted in MDL:
// https://github.com/AcademySoftwareFoundation/MaterialX/issues/1038
//
// We implement this patch on the MaterialX document level too, however we replicate it here so that
// HdMtlx does not throw validation errors due to mismatching NodeDefs.
void _PatchUsdPreviewSurfaceNormalColor3Vector3Mismatch(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;
    if (node.nodeTypeId != _tokens->ND_UsdPreviewSurface_surfaceshader)
    {
      continue;
    }

    auto& inputs = node.inputConnections;

    auto inputIt = inputs.find(_tokens->normal);
    if (inputIt == inputs.end())
    {
      return;
    }

    auto& connections = inputIt->second;
    for (HdMaterialConnection2& connection : connections)
    {
      if (connection.upstreamOutputName != _tokens->rgb)
      {
        continue;
      }

      SdfPath upstreamNodePath = connection.upstreamNode;

      SdfPath convertNodePath = upstreamNodePath;
      for (int i = 0; network.nodes.count(convertNodePath) > 0; i++)
      {
        std::string convertNodeName = "convert" + std::to_string(i);
        convertNodePath = upstreamNodePath.AppendElementString(convertNodeName);
      }

      HdMaterialNode2 convertNode;
      convertNode.nodeTypeId = _tokens->ND_convert_color3_vector3;
      convertNode.inputConnections[_tokens->in] = { { upstreamNodePath, _tokens->rgb } };
      network.nodes[convertNodePath] = convertNode;

      connection.upstreamNode = convertNodePath;
      connection.upstreamOutputName = _tokens->out;
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

  // Second pass: substitute node names and parameters
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
  : m_mtlxStdLib(mtlxStdLib)
{
}

GiMaterial* MaterialNetworkCompiler::CompileNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const
{
  GiMaterial* result = TryCompileMdlNetwork(network);

  if (!result)
  {
    HdMaterialNetwork2 patchedNetwork = network;

    PreviewSurfaceNetworkPatcher patcher;
    patcher.Patch(patchedNetwork);

    result = TryCompileMtlxNetwork(id, patchedNetwork);
  }

  return result;
}

GiMaterial* MaterialNetworkCompiler::TryCompileMdlNetwork(const HdMaterialNetwork2& network) const
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

  return giCreateMaterialFromMdlFile(fileUri.c_str(), subIdentifier.c_str());
}

GiMaterial* MaterialNetworkCompiler::TryCompileMtlxNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const
{
  HdMaterialNetwork2 mtlxNetwork = network;
  if (!_ConvertUsdNodesToMtlxNodes(mtlxNetwork))
  {
    return nullptr;
  }

  _PatchUsdPreviewSurfaceNormalColor3Vector3Mismatch(mtlxNetwork);

  mx::DocumentPtr doc = CreateMaterialXDocumentFromNetwork(id, mtlxNetwork);
  if (!doc)
  {
    return nullptr;
  }

  return giCreateMaterialFromMtlxDoc(doc);
}

mx::DocumentPtr MaterialNetworkCompiler::CreateMaterialXDocumentFromNetwork(const SdfPath& id,
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
    m_mtlxStdLib,
#if PXR_VERSION >= 2211
    &mxHdData
#else
    &hdTextureNodes,
    &mxHdTextureMap
#endif
  );
}

PXR_NAMESPACE_CLOSE_SCOPE
