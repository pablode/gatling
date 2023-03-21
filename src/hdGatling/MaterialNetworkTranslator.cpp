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

#include "MaterialNetworkTranslator.h"

#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXCore/Material.h>
#include <MaterialXCore/Definition.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <gi.h>

#include "MaterialNetworkPatcher.h"
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
  ((UsdUVTexture_wrapS, "wrapS"))
  ((UsdUVTexture_wrapT, "wrapT"))
  ((UsdUVTexture_WrapMode_black, "black"))
  ((UsdUVTexture_WrapMode_clamp, "clamp"))
  ((UsdUVTexture_WrapMode_repeat, "repeat"))
  ((UsdUVTexture_WrapMode_mirror, "mirror"))
  ((UsdUVTexture_WrapMode_useMetadata, "useMetdata"))
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
  ((ND_UsdUVTexture_WrapMode_black, "black"))
  ((ND_UsdUVTexture_WrapMode_clamp, "clamp"))
  ((ND_UsdUVTexture_WrapMode_periodic, "periodic"))
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

bool _ConvertUsdNodesToMaterialXNodes(const HdMaterialNetwork2& network,
                                      HdMaterialNetwork2& mtlxNetwork)
{
  mtlxNetwork = network;

  for (auto nodeIt = mtlxNetwork.nodes.begin(); nodeIt != mtlxNetwork.nodes.end(); nodeIt++)
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

        if (wrapToken == _tokens->UsdUVTexture_WrapMode_black)
        {
          // It's internally mapped to 'constant' which uses the fallback color
          TF_WARN("UsdUVTexture wrap mode black is not fully supported");
        }
        else if (wrapToken == _tokens->UsdUVTexture_WrapMode_mirror ||
                 wrapToken == _tokens->UsdUVTexture_WrapMode_clamp)
        {
          // These are valid, do nothing.
        }
        else if (wrapToken == _tokens->UsdUVTexture_WrapMode_repeat)
        {
          wrapType = _tokens->ND_UsdUVTexture_WrapMode_periodic;
        }
        else
        {
          TF_WARN("UsdUVTexture node has unsupported wrap mode %s\n", wrapToken.GetText());
          wrapType = _tokens->ND_UsdUVTexture_WrapMode_periodic;
        }
      };

      auto& parameters = nodeIt->second.parameters;
      auto wrapS = parameters.find(_tokens->UsdUVTexture_wrapS);
      auto wrapT = parameters.find(_tokens->UsdUVTexture_wrapT);

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

MaterialNetworkTranslator::MaterialNetworkTranslator(const std::string& mtlxLibPath)
{
  m_nodeLib = mx::createDocument();

  mx::FilePathVec libFolders; // All directories if left empty.
  mx::FileSearchPath folderSearchPath(mtlxLibPath);
  mx::loadLibraries(libFolders, folderSearchPath, m_nodeLib);
}

GiMaterial* MaterialNetworkTranslator::ParseNetwork(const SdfPath& id,
                                                     const HdMaterialNetwork2& network) const
{
  GiMaterial* result = TryParseMdlNetwork(network);

  if (!result)
  {
    result = TryParseMtlxNetwork(id, network);
  }

  return result;
}

GiMaterial* MaterialNetworkTranslator::TryParseMdlNetwork(const HdMaterialNetwork2& network) const
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

GiMaterial* MaterialNetworkTranslator::TryParseMtlxNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const
{
  HdMaterialNetwork2 mtlxNetwork;
  if (!_ConvertUsdNodesToMaterialXNodes(network, mtlxNetwork))
  {
    return nullptr;
  }

  MaterialNetworkPatcher patcher;
  patcher.Patch(mtlxNetwork);

  mx::DocumentPtr doc = CreateMaterialXDocumentFromNetwork(id, mtlxNetwork);
  if (!doc)
  {
    return nullptr;
  }

  mx::string docStr = mx::writeToXmlString(doc);

  return giCreateMaterialFromMtlxStr(docStr.c_str());
}

mx::DocumentPtr MaterialNetworkTranslator::CreateMaterialXDocumentFromNetwork(const SdfPath& id,
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
    m_nodeLib,
#if PXR_VERSION >= 2211
    &mxHdData
#else
    &hdTextureNodes,
    &mxHdTextureMap
#endif
  );
}

PXR_NAMESPACE_CLOSE_SCOPE
