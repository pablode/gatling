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

#include <string>
#include <vector>

namespace gtl
{
  constexpr const char* MC_MTLX_GTL_PARAMETER_NODE_CATEGORY = "GTL_parameter";

  class McMtlxDocumentParser
  {
  public:
    McMtlxDocumentParser(const MaterialX::DocumentPtr& stdLib);

    MaterialX::DocumentPtr parse(std::string_view str);

  private:
    MaterialX::DocumentPtr m_stdLib;
  };

  struct McMtlxNetworkDiff
  {
    MaterialX::InputPtr input;
    MaterialX::ValuePtr value;
  };

  struct McMtlxTextureParameter
  {
    std::string parameterName;
    std::string mdlParameterName;
    std::string filePath;
  };

  MaterialX::NodePtr McMtlxFindSurfaceShader(const MaterialX::DocumentPtr& doc, const char* name = nullptr);

  GbHash McMtlxHashNetworkTopological(const MaterialX::NodePtr& surfaceShader);

  std::vector<McMtlxNetworkDiff> McMtlxDiffTopoEqualNetworks(const MaterialX::NodePtr& referenceSurfaceShader,
                                                             const MaterialX::NodePtr& otherSurfaceShader);

  std::string McMtlxGetParameterName(const MaterialX::InputPtr& input);

  bool McMtlxInsertParameterNodes(const MaterialX::NodePtr& sourceNetwork,
                                  const std::vector<McMtlxNetworkDiff>& diff,
                                  MaterialX::DocumentPtr& outDoc,
                                  std::vector<McMtlxTextureParameter>& outTextureParameters);
}
