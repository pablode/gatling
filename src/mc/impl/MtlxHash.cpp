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
#include <assert.h>

namespace mx = MaterialX;

namespace gtl
{
  // FIXME: this function assumes no cycles
  McMtlxNodeHashMap McHashMtlxNetworkTopological(const MaterialX::NodePtr& surfaceShader)
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

  McMtlxTopoNetworkDiff McDiffTopoEquivalentMtlxNetworks(const MaterialX::NodePtr& surfaceShader1,
                                                         const MaterialX::NodePtr& surfaceShader2)
  {
    McMtlxTopoNetworkDiff diff;

    std::unordered_set<mx::NodePtr> visited;

    std::function<void(const mx::NodePtr&, const mx::NodePtr&)> traverseNode;

    traverseNode = [&](const mx::NodePtr& node1, const mx::NodePtr& node2)
    {
      assert(node1 && node2);

      if (visited.count(node1) > 0)
      {
        return;
      }

      mx::NodeDefPtr nodeDef = node1->getNodeDef();
      assert(nodeDef);

      auto ndInputs = nodeDef->getInputs();

      for (const mx::InputPtr& ndInput : ndInputs)
      {
        const std::string& inputName = ndInput->getName();

        mx::InputPtr input1 = node1->getInput(inputName);
        mx::InputPtr input2 = node2->getInput(inputName);

        if (input1)
        {
          assert(input2);
          const mx::NodePtr& upstreamNode1 = input1->getConnectedNode();

          if (upstreamNode1)
          {
            mx::NodePtr upstreamNode2 = input2->getConnectedNode();
            assert(upstreamNode2);

            traverseNode(upstreamNode1, upstreamNode2);
            continue;
          }
        }

        mx::ValuePtr value1 = input1 ? input1->getValue() : ndInput->getValue();
        mx::ValuePtr value2 = input2 ? input2->getValue() : ndInput->getValue();

        // NOTE: improved comparison func proposed in MaterialX PR #2199
        if (value1->getValueString() != value2->getValueString())
        {
          diff[node1].insert(inputName);
        }
      }

      visited.insert(node1);
    };

    assert(surfaceShader1 && surfaceShader2);

    traverseNode(surfaceShader1, surfaceShader2);

    return diff;
  }
}
