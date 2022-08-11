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

  // Workaround for HdMtlxConvertToString not handling the TfToken type:
  // https://github.com/PixarAnimationStudios/USD/blob/3abc46452b1271df7650e9948fef9f0ce602e3b2/pxr/imaging/hdMtlx/hdMtlx.cpp#L117
  class TfTokenPatcher : public PatcherBase
  {
  public:
    void PatchNode(HdMaterialNode2& node) override
    {
      auto& parameters = node.parameters;

      for (auto& tokenValuePair : parameters)
      {
        VtValue& value = tokenValuePair.second;

        if (value.IsHolding<TfToken>())
        {
          value = value.Cast<std::string>();
        }
      }
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

  patchers.push_back(std::make_unique<detail::TfTokenPatcher>());

  for (auto& patcher : patchers)
  {
    patcher->PatchNetwork(network);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
