#include "MdlParserPlugin.h"

#include <pxr/usd/sdr/shaderNode.h>
#include <pxr/usd/ar/resolver.h>
#include "pxr/usd/ar/resolvedPath.h"
#include "pxr/usd/ar/asset.h"
#include <pxr/usd/ar/ar.h>

#include "Tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

NdrNodeUniquePtr HdGatlingMdlParserPlugin::Parse(const NdrNodeDiscoveryResult& discoveryResult)
{
  NdrTokenMap metadata = discoveryResult.metadata;
  metadata[HdGatlingNodeMetadata->subIdentifier] = discoveryResult.subIdentifier;

  return std::make_unique<SdrShaderNode>(
    discoveryResult.identifier,
    discoveryResult.version,
    discoveryResult.name,
    discoveryResult.family,
    HdGatlingNodeContexts->mdl,
    discoveryResult.sourceType,
    discoveryResult.uri,
    discoveryResult.resolvedUri,
    NdrPropertyUniquePtrVec{},
    metadata
  );
}

const NdrTokenVec& HdGatlingMdlParserPlugin::GetDiscoveryTypes() const
{
  static NdrTokenVec s_discoveryTypes{ HdGatlingDiscoveryTypes->mdl };
  return s_discoveryTypes;
}

const TfToken& HdGatlingMdlParserPlugin::GetSourceType() const
{
  return HdGatlingSourceTypes->mdl;
}

NDR_REGISTER_PARSER_PLUGIN(HdGatlingMdlParserPlugin);

PXR_NAMESPACE_CLOSE_SCOPE
