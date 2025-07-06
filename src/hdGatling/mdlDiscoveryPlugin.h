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

#pragma once

#include <pxr/pxr.h>
#if PXR_VERSION >= 2508
#include <pxr/usd/sdr/discoveryPlugin.h>
#else
#include <pxr/usd/ndr/discoveryPlugin.h>
#endif

PXR_NAMESPACE_OPEN_SCOPE

#if PXR_VERSION >= 2508
class HdGatlingMdlDiscoveryPlugin final : public SdrDiscoveryPlugin
{
public:
  SdrShaderNodeDiscoveryResultVec DiscoverShaderNodes(const Context& ctx) override;

  const SdrStringVec& GetSearchURIs() const override;
};
#else
class HdGatlingMdlDiscoveryPlugin final : public NdrDiscoveryPlugin
{
public:
  NdrNodeDiscoveryResultVec DiscoverNodes(const Context& ctx) override;

  const NdrStringVec& GetSearchURIs() const override;
};
#endif

PXR_NAMESPACE_CLOSE_SCOPE
