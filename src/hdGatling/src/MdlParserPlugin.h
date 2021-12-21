#pragma once

#include <pxr/usd/ndr/parserPlugin.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingMdlParserPlugin final : public NdrParserPlugin
{
public:
  NdrNodeUniquePtr Parse(const NdrNodeDiscoveryResult& discoveryResult) override;

  const NdrTokenVec& GetDiscoveryTypes() const override;

  const TfToken& GetSourceType() const override;
};

PXR_NAMESPACE_CLOSE_SCOPE
