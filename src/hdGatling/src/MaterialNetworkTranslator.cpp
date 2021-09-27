#include "MaterialNetworkTranslator.h"

#include <pxr/imaging/hd/material.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXCore/Material.h>
#include <MaterialXCore/Definition.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <gi.h>

#include "Tokens.h"

namespace mx = MaterialX;

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  // USD node types
  (UsdPreviewSurface)
  (UsdUVTexture)
  (UsdTransform2d)
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
  // MaterialX UsdPreviewSurface node types
  (ND_UsdPreviewSurface_surfaceshader)
  (ND_UsdUVTexture)
  (ND_UsdPrimvarReader_integer)
  (ND_UsdPrimvarReader_boolean)
  (ND_UsdPrimvarReader_string)
  (ND_UsdPrimvarReader_float)
  (ND_UsdPrimvarReader_vector2)
  (ND_UsdPrimvarReader_vector3)
  (ND_UsdPrimvarReader_vector4)
  (ND_UsdTransform2d)
  (ND_UsdPrimvarReader_matrix44)
);

bool _ConvertNodesToMaterialXNodes(const HdMaterialNetwork2& network,
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

    if (nodeTypeId == _tokens->UsdPreviewSurface)
    {
      nodeTypeId = _tokens->ND_UsdPreviewSurface_surfaceshader;
    }
    else if (nodeTypeId == _tokens->UsdUVTexture)
    {
      nodeTypeId = _tokens->ND_UsdUVTexture;
    }
    else if (nodeTypeId == _tokens->UsdTransform2d)
    {
      nodeTypeId = _tokens->ND_UsdTransform2d;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_float)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_float;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_float2)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_vector2;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_float3)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_vector3;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_float4)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_vector4;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_int)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_integer;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_string)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_string;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_normal)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_vector3;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_point)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_vector3;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_vector)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_vector3;
    }
    else if (nodeTypeId == _tokens->UsdPrimvarReader_matrix)
    {
      nodeTypeId = _tokens->ND_UsdPrimvarReader_matrix44;
    }
    else
    {
      TF_WARN("Unable to translate material node of type %s to MaterialX counterpart", nodeTypeId.GetText());
      return false;
    }
  }

  return true;
}

bool _GetMaterialNetworkSurfaceTerminal(const HdMaterialNetwork2& network2, HdMaterialNode2& surfaceTerminal)
{
  const auto& connectionIt = network2.terminals.find(HdMaterialTerminalTokens->surface);

  if (connectionIt == network2.terminals.end())
  {
    return false;
  }

  const HdMaterialConnection2& connection = connectionIt->second;

  const SdfPath& terminalPath = connection.upstreamNode;

  const auto& nodeIt = network2.nodes.find(terminalPath);

  if (nodeIt == network2.nodes.end())
  {
    return false;
  }

  surfaceTerminal = nodeIt->second;

  return true;
}

MaterialNetworkTranslator::MaterialNetworkTranslator(const std::string& mtlxLibPath)
{
  m_nodeLib = mx::createDocument();

  mx::FilePathVec libFolders; // All directories if left empty.
  mx::FileSearchPath folderSearchPath(mtlxLibPath);
  mx::loadLibraries(libFolders, folderSearchPath, m_nodeLib);
}

gi_material* MaterialNetworkTranslator::ParseNetwork(const SdfPath& id,
                                                     const HdMaterialNetwork2& network) const
{
  HdMaterialNetwork2 mtlxNetwork;
  if (!_ConvertNodesToMaterialXNodes(network, mtlxNetwork))
  {
    return nullptr;
  }

  mx::DocumentPtr doc = CreateMaterialXDocumentFromNetwork(id, mtlxNetwork);
  if (!doc)
  {
    return nullptr;
  }

  mx::string docStr = mx::writeToXmlString(doc);

  return giCreateMaterialFromMtlx(docStr.c_str());
}

mx::DocumentPtr MaterialNetworkTranslator::CreateMaterialXDocumentFromNetwork(const SdfPath& id,
                                                                              const HdMaterialNetwork2& network) const
{
  HdMaterialNode2 surfaceTerminal;
  if (!_GetMaterialNetworkSurfaceTerminal(network, surfaceTerminal))
  {
    TF_WARN("Unable to find surface terminal for material network");
    return nullptr;
  }

  std::set<SdfPath> hdTextureNodes;
  mx::StringMap mxHdTextureMap;

  return HdMtlxCreateMtlxDocumentFromHdNetwork(
    network,
    surfaceTerminal,
    id,
    m_nodeLib,
    &hdTextureNodes,
    &mxHdTextureMap
  );
}

PXR_NAMESPACE_CLOSE_SCOPE
