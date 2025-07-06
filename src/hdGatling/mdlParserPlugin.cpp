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

#include "mdlParserPlugin.h"
#include "tokens.h"

#include <pxr/usd/sdr/shaderNode.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/ar.h>

PXR_NAMESPACE_OPEN_SCOPE

#if PXR_VERSION >= 2508
SdrShaderNodeUniquePtr HdGatlingMdlParserPlugin::ParseShaderNode(const SdrShaderNodeDiscoveryResult& discoveryResult)
{
  SdrTokenMap metadata = discoveryResult.metadata;
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
    SdrShaderPropertyUniquePtrVec{},
    metadata
  );
}

const SdrTokenVec& HdGatlingMdlParserPlugin::GetDiscoveryTypes() const
{
  static SdrTokenVec s_discoveryTypes{ HdGatlingDiscoveryTypes->mdl };
  return s_discoveryTypes;
}

#else

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
#endif

const TfToken& HdGatlingMdlParserPlugin::GetSourceType() const
{
  return HdGatlingSourceTypes->mdl;
}

#if PXR_VERSION >= 2508
SDR_REGISTER_PARSER_PLUGIN(HdGatlingMdlParserPlugin);
#else
NDR_REGISTER_PARSER_PLUGIN(HdGatlingMdlParserPlugin);
#endif

PXR_NAMESPACE_CLOSE_SCOPE
