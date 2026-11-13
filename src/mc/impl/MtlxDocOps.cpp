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

#include "MtlxDocOps.h"

#include <MaterialXFormat/Util.h>
#include <MaterialXGenShader/Util.h>

#include <gtl/gb/Log.h>

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mx = MaterialX;

namespace
{
  std::string _GetGraphOutputSelection(const mx::OutputPtr& graphOutput,
                                       const mx::NodePtr& outputNode)
  {
    std::string selection = graphOutput->getOutputString();

    if (selection.empty() && outputNode && !outputNode->getOutputs().empty())
    {
      selection = outputNode->getOutputs()[0]->getName();
    }

    return selection;
  }

  mx::InputPtr _ResolveInterfaceInput(const mx::InputPtr& input)
  {
    mx::InputPtr resolvedInput = input;
    std::unordered_set<mx::InputPtr> visitedInputs;

    while (mx::InputPtr interfaceInput = resolvedInput->getInterfaceInput())
    {
      if (!visitedInputs.insert(interfaceInput).second)
      {
        break;
      }
      resolvedInput = interfaceInput;
    }

    return resolvedInput;
  }
}

namespace gtl
{
  McMtlxDocumentParser::McMtlxDocumentParser(const mx::DocumentPtr& stdLib)
    : m_stdLib(stdLib)
  {
  }

  mx::DocumentPtr McMtlxDocumentParser::parse(std::string_view str)
  {
    try
    {
      mx::DocumentPtr doc = mx::createDocument();
      doc->importLibrary(m_stdLib);
      mx::readFromXmlString(doc, str.data());
      return doc;
    }
    catch (const std::exception& ex)
    {
      return nullptr;;
    }
  }

  mx::NodePtr McMtlxFindSurfaceShader(const mx::DocumentPtr& doc, const char* name)
  {
    // Find material in document
    auto materialNodes = doc->getMaterialNodes();

    if (materialNodes.size() != 1)
    {
      // HdMtlx generates a new document for each material; there should be a single material
      GB_ERROR("unexpected material count in mtlx document for material {}", name ? name : "<unnamed>");
      return nullptr;
    }

    mx::NodePtr materialNode = materialNodes[0];

    // Extract surface shader node
    auto surfaceShaderNodes = mx::getShaderNodes(materialNode, mx::SURFACE_SHADER_TYPE_STRING);

    mx::TypedElementPtr renderableElement = materialNode;
    if (!surfaceShaderNodes.empty())
    {
      renderableElement = *surfaceShaderNodes.begin();

      if (surfaceShaderNodes.size() > 1)
      {
        GB_WARN("found multiple surface shaders for material {}; choosing first", name ? name : "<unnamed>");
      }
    }

    mx::ElementPtr surfaceElement = doc->getDescendant(renderableElement->getNamePath());
    if (!surfaceElement)
    {
      GB_WARN("could not find surface shader for material {}", name ? name : "<unnamed>");
      return nullptr;
    }

    GB_DEBUG("found surface shader: {}", surfaceElement->getName());
    return surfaceElement->asA<mx::Node>();
  }

  GbHash McMtlxHashNetworkTopological(const MaterialX::NodePtr& surfaceShader)
  {
    std::unordered_map<mx::NodePtr, GbHash> topoHashes;
    std::unordered_set<mx::NodePtr> nodesInProgress;

    // We don't use topolical sorting, but instead traverse the graph. this culls nodes
    // from the document and also disregards node graph boundaries.

    std::function<GbHash(const mx::NodePtr& node)> hashNode;

    hashNode = [&](const mx::NodePtr& node)
    {
      auto topoHashIt = topoHashes.find(node);
      if (topoHashIt != topoHashes.end())
      {
        return topoHashIt->second;
      }

      if (nodesInProgress.count(node) > 0)
      {
        return GbHashAppend(GbHash(), node->getName());
      }

      nodesInProgress.insert(node);

      GbHash hash;

      mx::NodeDefPtr nodeDef = node->getNodeDef();
      hash = GbHashAppend(hash, nodeDef ? nodeDef->getName() : node->getCategory());
      hash = GbHashAppend(hash, node->getType());

      // Increase chance of deduplication with deterministic sort order
      std::vector<mx::InputPtr> inputs = node->getActiveInputs();
      std::sort(inputs.begin(), inputs.end(), [](const mx::InputPtr& a, const mx::InputPtr& b)
      {
        return a->getName() < b->getName();
      });

      for (mx::InputPtr input : inputs)
      {
        mx::InputPtr connectionInput = _ResolveInterfaceInput(input);

        hash = GbHashAppend(hash, input->getName());
        hash = GbHashAppend(hash, input->getType());

        if (connectionInput->hasColorSpace())
        {
          hash = GbHashAppend(hash, connectionInput->getColorSpace());
        }
        if (connectionInput->hasUnit())
        {
          hash = GbHashAppend(hash, connectionInput->getUnit());
        }

        mx::NodePtr upstreamNode = connectionInput->getConnectedNode();
        mx::OutputPtr upstreamOutput = connectionInput->getConnectedOutput();

        if (upstreamOutput)
        {
          mx::ElementPtr outputParent = upstreamOutput->getParent();

          if (outputParent && outputParent->isA<mx::NodeGraph>())
          {
            hash = GbHashAppend(hash, _GetGraphOutputSelection(upstreamOutput, upstreamNode));
          }
          else
          {
            hash = GbHashAppend(hash, upstreamOutput->getName());
          }
        }

        if (upstreamNode)
        {
          hash = GbHashCombine(hash, hashNode(upstreamNode));
        }
      }

      if (node->hasColorSpace())
      {
        hash = GbHashAppend(hash, node->getColorSpace());
      }

      topoHashes[node] = hash;
      nodesInProgress.erase(node);

      return hash;
    };

    return hashNode(surfaceShader);
  }

  std::vector<McMtlxNetworkDiff> McMtlxDiffTopoEqualNetworks(const mx::NodePtr& referenceSurfaceShader,
                                                             const mx::NodePtr& otherSurfaceShader)
  {
    std::vector<McMtlxNetworkDiff> diff;
    std::unordered_set<mx::NodePtr> visitedNodes;

    std::function<void(const mx::NodePtr&, const mx::NodePtr&)> diffNode;

    diffNode = [&](const mx::NodePtr& referenceNode, const mx::NodePtr& otherNode)
    {
      if (!visitedNodes.insert(referenceNode).second)
      {
        return;
      }

      std::vector<mx::InputPtr> referenceInputs = referenceNode->getActiveInputs();
      std::sort(referenceInputs.begin(), referenceInputs.end(), [](const mx::InputPtr& a, const mx::InputPtr& b)
      {
        return a->getName() < b->getName();
      });

      for (const mx::InputPtr& referenceInput : referenceInputs)
      {
        mx::InputPtr otherInput = otherNode->getActiveInput(referenceInput->getName());
        if (!otherInput)
        {
          continue;
        }

        mx::InputPtr referenceConnection = _ResolveInterfaceInput(referenceInput);
        mx::InputPtr otherConnection = _ResolveInterfaceInput(otherInput);

        mx::NodePtr referenceUpstreamNode = referenceConnection->getConnectedNode();
        mx::NodePtr otherUpstreamNode = otherConnection->getConnectedNode();

        if (referenceUpstreamNode && otherUpstreamNode)
        {
          diffNode(referenceUpstreamNode, otherUpstreamNode);
          continue;
        }

        mx::ValuePtr referenceValue = referenceConnection->getValue();
        mx::ValuePtr otherValue = otherConnection->getValue();

        bool valuesEqual = (!referenceValue && !otherValue) ||
                           (referenceValue && otherValue && referenceValue->isEqual(otherValue));

        if (!valuesEqual)
        {
          diff.push_back({ referenceConnection, otherValue });
        }
      }
    };

    diffNode(referenceSurfaceShader, otherSurfaceShader);

    return diff;
  }
}
