#include "MdlDiscoveryPlugin.h"

#include <pxr/base/tf/staticTokens.h>

#include "Tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

NdrNodeDiscoveryResultVec HdGatlingMdlDiscoveryPlugin::DiscoverNodes(const Context& ctx)
{
  NdrNodeDiscoveryResultVec result;

  NdrNodeDiscoveryResult mdlNode(
    HdGatlingNodeIdentifiers->mdl, // identifier
    NdrVersion(1),                 // version
    HdGatlingNodeIdentifiers->mdl, // name
    TfToken(),                     // family
    HdGatlingDiscoveryTypes->mdl,  // discoveryType
    HdGatlingSourceTypes->mdl,     // sourceType
    std::string(),                 // uri
    std::string()                  // resolvedUri
  );
  result.push_back(mdlNode);

  return result;
}

const NdrStringVec& HdGatlingMdlDiscoveryPlugin::GetSearchURIs() const
{
  static const NdrStringVec s_searchURIs;
  return s_searchURIs;
}

NDR_REGISTER_DISCOVERY_PLUGIN(HdGatlingMdlDiscoveryPlugin);

PXR_NAMESPACE_CLOSE_SCOPE
