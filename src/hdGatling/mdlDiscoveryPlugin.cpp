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

#include "mdlDiscoveryPlugin.h"
#include "tokens.h"

#include <pxr/base/tf/staticTokens.h>

PXR_NAMESPACE_OPEN_SCOPE

NdrNodeDiscoveryResultVec HdGatlingMdlDiscoveryPlugin::DiscoverNodes([[maybe_unused]] const Context& ctx)
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
