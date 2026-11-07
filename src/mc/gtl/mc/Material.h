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

#include <stdint.h>
#include <string>
#include <memory>
#include <variant>
#include <vector>

namespace gtl
{
  struct McMdlMaterial;

  enum class McOpacityClassification
  {
    Opaque,
    Cutout,
    NonBinary
  };

  // Categories of the custom nodes which replace parameterized inputs in the material network.
  constexpr const char* MC_MTLX_SCALAR_PARAMETER_NODE_CATEGORY = "GTL_scalar_parameter";
  constexpr const char* MC_MTLX_TEXTURE_FILE_NODE_CATEGORY = "GTL_texture_file";

  struct McMtlxScalarNetworkParameter
  {
    std::string name;
    std::vector<uint8_t> data;
    uint32_t componentCount;
  };

  struct McMtlxTextureNetworkParameter
  {
    std::string name;
    std::string mdlParameterName;
    std::string uri;
    std::string filePath;
  };

  using McMtlxNetworkParameter = std::variant<McMtlxScalarNetworkParameter, McMtlxTextureNetworkParameter>;

  inline const std::string& McMtlxGetNetworkParameterName(const McMtlxNetworkParameter& parameter)
  {
    return std::visit([](const auto& p) -> const std::string& { return p.name; }, parameter);
  }

  struct McMaterial
  {
    bool hasBackfaceBsdf;
    bool hasBackfaceEdf;
    bool hasVolumeAbsorptionCoeff;
    bool hasVolumeScatteringCoeff;
    McOpacityClassification opacityClass;
    bool isAnimated;
    bool isEmissive;
    bool isThinWalled;
    float directionalBias;
    std::string resourcePathPrefix;
    std::shared_ptr<McMdlMaterial> mdlMaterial;
    bool requiresSceneTransforms;
    std::vector<const char*> sceneDataNames;
  };
}
