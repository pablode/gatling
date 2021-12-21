#include "MdlDiscoveryPlugin.h"

#include <pxr/base/tf/staticTokens.h>

#include "Tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

NdrNodeDiscoveryResultVec HdGatlingMdlDiscoveryPlugin::DiscoverNodes(const Context& ctx)
{
  NdrNodeDiscoveryResultVec result;

  NdrNodeDiscoveryResult mdlNode(
    /* identifier    */ HdGatlingNodeIdentifiers->mdl,
    /* version       */ NdrVersion(1),
    /* name          */ HdGatlingNodeIdentifiers->mdl,
    /* family        */ TfToken(),
    /* discoveryType */ HdGatlingDiscoveryTypes->mdl,
    /* sourceType    */ HdGatlingSourceTypes->mdl,
    /* uri           */ std::string(),
    /* resolvedUri   */ std::string()
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
