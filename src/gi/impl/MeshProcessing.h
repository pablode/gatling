//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#include <vector>

#include <Gi.h>

namespace gtl
{
  struct GiMeshBuffer
  {
    bool isCompressed;
    uint32_t uncompressedSize;
    std::vector<uint8_t> data;
  };

  struct GiMeshPrimvar
  {
    std::string name;
    GiPrimvarType type;
    GiPrimvarInterpolation interpolation;
    GiMeshBuffer buffer;
  };

  struct GiMeshData
  {
    GiMeshBuffer faces;
    GiMeshBuffer vertices;
    std::vector<GiMeshPrimvar> primvars;
    uint32_t faceCount;
    uint32_t vertexCount;
  };

  GiMeshData giProcessMeshData(const std::vector<GiFace>& faces,
                               const std::vector<GiVertex>& vertices,
                               const std::vector<GiPrimvarData>& primvars);

  void giDecompressMeshData(const GiMeshData& cmd,
                            std::vector<GiFace>& faces,
                            std::vector<GiVertex>& vertices,
                            std::vector<GiPrimvarData>& primvars);
}
