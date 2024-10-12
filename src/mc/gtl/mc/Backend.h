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

  enum class McDfFlags
  {
    Scattering                = (1 <<  0),
    Emission                  = (1 <<  1),
    EmissionIntensity         = (1 <<  2),
    ThinWalled                = (1 <<  3),
    VolumeAbsorption          = (1 <<  4),
    VolumeScattering          = (1 <<  5),
    CutoutOpacity             = (1 <<  6),
    Ior                       = (1 <<  7),
    BackfaceScattering        = (1 <<  8),
    BackfaceEmission          = (1 <<  9),
    BackfaceEmissionIntensity = (1 << 10),
    FLAG_COUNT                = 11
  };
  GB_DECLARE_ENUM_BITOPS(McDfFlags)

  class McBackend
  {
  public:
    bool init(McRuntime& runtime);

    bool genGlsl(const McMdlMaterial& material,
                 McDfFlags dfFlags,
                 McGlslGenResult& result);

  private:
    class _Impl;
    std::shared_ptr<_Impl> m_impl;
  };
}
