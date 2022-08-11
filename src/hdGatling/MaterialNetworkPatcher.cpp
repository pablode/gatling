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

#include "MaterialNetworkPatcher.h"

#include <pxr/imaging/hd/material.h>

#include <memory>

namespace detail
{
  using namespace PXR_NS;

  class PatcherBase
  {
  public:
    virtual void PatchNetwork(HdMaterialNetwork2& network)
    {
      for (auto& pathNodePair : network.nodes)
      {
        PatchNode(pathNodePair.second);
      }
    }

    virtual void PatchNode(HdMaterialNode2& node)
    {
    }
  };
}

PXR_NAMESPACE_OPEN_SCOPE

MaterialNetworkPatcher::MaterialNetworkPatcher()
{
}

void MaterialNetworkPatcher::Patch(HdMaterialNetwork2& network)
{
  using PatcherBasePtr = std::unique_ptr<detail::PatcherBase>;

  std::vector<PatcherBasePtr> patchers;

  // TODO: add patchers

  for (auto& patcher : patchers)
  {
    patcher->PatchNetwork(network);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
