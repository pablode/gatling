//
// Copyright (C) 2025 Pablo Delgado Krämer
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

#include "MtlxHash.h"

#include <MaterialXCore/Document.h>

#include <gtl/gb/Hash.h>

#include <unordered_map>

namespace mx = MaterialX;

namespace gtl
{
  // FIXME: this function assumes no cycles
  McMtlxNodeHashMap McHashMtlxNetworkTopological(const MaterialX::DocumentPtr& doc,
                                                 const MaterialX::NodePtr& surfaceShader)
  {
    std::unordered_map<mx::NodePtr, GbHash> hashes;

    // We don't use topolical sorting, but instead traverse the graph. this culls nodes
    // from the document and also disregards node graph boundaries.

    std::function<GbHash(const mx::NodePtr&)> hashNode;

    hashNode = [&](const mx::NodePtr& node)
    {
      if (hashes.count(node) > 0)
      {
        return hashes[node];
      }

      GbHash hash = 0;

      mx::NodeDefPtr nodeDef = node->getNodeDef();
      hash = GbHashAppend(hash, nodeDef->getName());

      for (mx::InputPtr input : node->getActiveInputs())
      {
        hash = GbHashAppend(hash, input->getName());

        mx::NodePtr upstreamNode = input->getConnectedNode();
        if (upstreamNode)
        {
          hash = GbHashCombine(hash, hashNode(upstreamNode));
        }

        if (input->hasColorSpace())
        {
          hash = GbHashAppend(hash, input->getColorSpace());
        }
      }

      if (node->hasColorSpace())
      {
        hash = GbHashAppend(hash, node->getColorSpace());
      }

      hashes[node] = hash;

      return hash;
    };

    hashNode(surfaceShader);

    return hashes;
  }
}
