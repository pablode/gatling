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
#include <vector>
#include <memory>
#include <unordered_map>

#include <gtl/gb/Enum.h>

namespace gtl
{
  class McRuntime;
  struct McMdlMaterial;

  struct McTextureDescription
  {
    uint32_t binding;
    bool is3dImage;
    bool isFloat;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    std::vector<uint8_t> data;
    std::string filePath;
  };

  struct McGlslGenResult
  {
    std::string source;
    std::vector<McTextureDescription> textureDescriptions;
  };

  enum class McDf
  {
    Scattering,
    Emission,
    EmissionIntensity,
    ThinWalled,
    VolumeAbsorption,
    VolumeScattering,
    CutoutOpacity,
    Ior,
    BackfaceScattering,
    BackfaceEmission,
    BackfaceEmissionIntensity,
    COUNT
  };

  using McDfMap = std::unordered_map<McDf, const char*>;

  class McBackend
  {
  public:
    bool init(McRuntime& runtime);

    void setAuxiliaryOutputEnabled(bool enabled);

    bool genGlsl(const McMdlMaterial& material, McDfMap dfMap, McGlslGenResult& result);

  private:
    class _Impl;
    std::shared_ptr<_Impl> m_impl;
  };
}
