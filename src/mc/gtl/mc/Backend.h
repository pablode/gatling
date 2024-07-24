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

  enum McDfFlags
  {
    MC_DF_FLAG_SCATTERING                  = (1 <<  0),
    MC_DF_FLAG_EMISSION                    = (1 <<  1),
    MC_DF_FLAG_EMISSION_INTENSITY          = (1 <<  2),
    MC_DF_FLAG_THIN_WALLED                 = (1 <<  3),
    MC_DF_FLAG_VOLUME_ABSORPTION           = (1 <<  4),
    MC_DF_FLAG_VOLUME_SCATTERING           = (1 <<  5),
    MC_DF_FLAG_CUTOUT_OPACITY              = (1 <<  6),
    MC_DF_FLAG_IOR                         = (1 <<  7),
    MC_DF_FLAG_BACKFACE_SCATTERING         = (1 <<  8),
    MC_DF_FLAG_BACKFACE_EMISSION           = (1 <<  9),
    MC_DF_FLAG_BACKFACE_EMISSION_INTENSITY = (1 << 10),
    MC_DF_FLAG_COUNT                       = 11
  };

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
