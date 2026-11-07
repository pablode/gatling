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

#include "Hash.h"

#include <MaterialXCore/Document.h>

#include <gtl/gb/Hash.h>
#include <gtl/mc/Material.h>

#include <unordered_map>

namespace mx = MaterialX;

namespace gtl
{
  // FIXME: this function assumes no cycles
  GbHash McHashMtlxNetworkTopological(const MaterialX::DocumentPtr& doc,
                                      const MaterialX::NodePtr& surfaceShader)
  {
    std::unordered_map<mx::NodePtr, GbHash> topoHashes;

    // We don't use topolical sorting, but instead traverse the graph. this culls nodes
    // from the document and also disregards node graph boundaries.

    std::function<GbHash(const mx::NodePtr& node)> hashNode;

    hashNode = [&](const mx::NodePtr& node)
    {
      if (topoHashes.count(node) > 0)
      {
        return topoHashes[node];
      }

      GbHash hash;

      // Resolves which output of the connected node a node graph output selects. An empty
      // selection means the node's first output, which must hash like an explicit selection.
      const auto appendGraphOutputSelection = [&hash](const mx::OutputPtr& graphOutput,
                                                      const mx::NodePtr& outputNode)
      {
        std::string selection = graphOutput->getOutputString();

        if (selection.empty() && outputNode && !outputNode->getOutputs().empty())
        {
          selection = outputNode->getOutputs()[0]->getName();
        }

        hash = GbHashAppend(hash, selection);
      };

      mx::NodeDefPtr nodeDef = node->getNodeDef();
      hash = GbHashAppend(hash, nodeDef ? nodeDef->getName() : node->getCategory());
      hash = GbHashAppend(hash, node->getType());

      for (mx::InputPtr input : node->getActiveInputs())
      {
        hash = GbHashAppend(hash, input->getName());
        hash = GbHashAppend(hash, input->getType());

        if (input->hasUnit())
        {
          hash = GbHashAppend(hash, input->getUnit());
        }

        mx::NodePtr upstreamNode = input->getConnectedNode();
        if (upstreamNode)
        {
          // The selected output is part of the network identity because nodes can have
          // multiple outputs with different values. For connections through node graphs, the
          // graph output's own selection is significant instead of its name.
          mx::OutputPtr upstreamOutput = input->getConnectedOutput();
          if (upstreamOutput)
          {
            if (input->hasNodeGraphString())
            {
              appendGraphOutputSelection(upstreamOutput, upstreamNode);
            }
            else
            {
              hash = GbHashAppend(hash, upstreamOutput->getName());
            }
          }

          hash = GbHashCombine(hash, hashNode(upstreamNode));
        }
        else if (mx::OutputPtr upstreamOutput = input->getConnectedOutput())
        {
          // Follow node graph outputs so that their contents contribute to the hash. Otherwise
          // materials with different graphs but identical outer networks would be deduplicated.
          mx::NodePtr outputNode = upstreamOutput->getConnectedNode();
          appendGraphOutputSelection(upstreamOutput, outputNode);
          if (outputNode)
          {
            hash = GbHashCombine(hash, hashNode(outputNode));
          }
        }
        else if (input->hasValue())
        {
          // Parameterized values are removed from the document and therefore do not
          // contribute to the hash. Remaining values (mostly strings and filenames)
          // are part of the network identity.
          if (node->getCategory() == MC_MTLX_TEXTURE_FILE_NODE_CATEGORY ||
              node->getCategory() == MC_MTLX_SCALAR_PARAMETER_NODE_CATEGORY)
          {
            // Parameter nodes only carry the parameter name and texture file path,
            // which are not part of the topology.
            continue;
          }

          hash = GbHashAppend(hash, input->getValueString());
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

      topoHashes[node] = hash;

      return hash;
    };

    return hashNode(surfaceShader);
  }
}
