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
#include <pxr/usd/sdr/parserPlugin.h>
#else
#include <pxr/usd/ndr/parserPlugin.h>
#endif

PXR_NAMESPACE_OPEN_SCOPE

#if PXR_VERSION >= 2508
class HdGatlingMdlParserPlugin final : public SdrParserPlugin
{
public:
  SdrShaderNodeUniquePtr ParseShaderNode(const SdrShaderNodeDiscoveryResult& discoveryResult) override;

  const SdrTokenVec& GetDiscoveryTypes() const override;

  const TfToken& GetSourceType() const override;
};
#else
class HdGatlingMdlParserPlugin final : public NdrParserPlugin
{
public:
  NdrNodeUniquePtr Parse(const NdrNodeDiscoveryResult& discoveryResult) override;

  const NdrTokenVec& GetDiscoveryTypes() const override;

  const TfToken& GetSourceType() const override;
};
#endif

PXR_NAMESPACE_CLOSE_SCOPE
