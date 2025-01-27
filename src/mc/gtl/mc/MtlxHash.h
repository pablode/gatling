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

#pragma once

#include <MaterialXCore/Document.h>

#include <gtl/gb/Hash.h>

#include <unordered_map>
#include <unordered_set>

// TODO: consistent McMtlx prefix. in general, rename functions.
namespace gtl
{
  using McMtlxNodeHashMap = std::unordered_map<MaterialX::NodePtr, GbHash>;
  using McMtlxTopoNetworkDiff = std::unordered_map<GbHash, std::unordered_set<std::string/* input names*/>>;
  using McMtlxNetworkValueDiff = std::vector<std::vector<MaterialX::ConstValuePtr>>;

  McMtlxNodeHashMap McHashMtlxNetworkTopological(const MaterialX::NodePtr& surfaceShader);

  McMtlxTopoNetworkDiff McDiffTopoEquivalentMtlxNetworks(const MaterialX::NodePtr& surfaceShader1,
                                                         const McMtlxNodeHashMap& nodeHashMap1,
                                                         const MaterialX::NodePtr& surfaceShader2);

  McMtlxNetworkValueDiff McMtlxExtractNetworkValues(const MaterialX::NodePtr& surfaceShader,
                                                    const McMtlxNodeHashMap& nodeHashMap,
                                                    const McMtlxTopoNetworkDiff& diff);

  // TODO: is it not clear yet where to write mx::ValuePtr into raw memory.

  // TODO: maybe MtlxDocUtils.h -> McBytePackMtlxValues(size_t alignment)

  // TODO: we also need another function that inserts geompropvalue nodes into a network.
  // TODO: this function should take McMtlxTopoNetworkDiff (<node, input>[]). maybe geomprop name prefix
}
