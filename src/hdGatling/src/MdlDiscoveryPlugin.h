#pragma once

#include <pxr/usd/ndr/discoveryPlugin.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingMdlDiscoveryPlugin final : public NdrDiscoveryPlugin
{
public:
  NdrNodeDiscoveryResultVec DiscoverNodes(const Context& ctx) override;

  const NdrStringVec& GetSearchURIs() const override;
};

PXR_NAMESPACE_CLOSE_SCOPE
