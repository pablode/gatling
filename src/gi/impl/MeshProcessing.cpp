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

#include "MeshProcessing.h"

#include <meshoptimizer.h>

#include <gtl/gb/Log.h>

using namespace gtl;

namespace
{
  uint32_t _PrimvarTypeSize(GiPrimvarType type)
  {
    switch (type)
    {
    case GiPrimvarType::Float:
    case GiPrimvarType::Int:
      return 4;

    case GiPrimvarType::Int2:
    case GiPrimvarType::Vec2:
      return 2 * 4;

    case GiPrimvarType::Int3:
    case GiPrimvarType::Vec3:
      return 3 * 4;

    case GiPrimvarType::Int4:
    case GiPrimvarType::Vec4:
      return 4 * 4;

    default:
      assert(false);
      GB_ERROR("coding error: unhandled type size!");
      return 0;
    }
  }
}

namespace gtl
{
  GiMeshData giProcessMeshData(uint32_t vertexCount, const GiVertex* vertices,
                               uint32_t faceCount, const GiFace* faces,
                               const std::vector<GiPrimvarData>& primvars)
  {
    std::vector<meshopt_Stream> streams;
    streams.reserve(primvars.size());

    streams.push_back(meshopt_Stream{ (void*) &vertices[0], sizeof(GiVertex), sizeof(GiVertex) });
    for (const GiPrimvarData& p : primvars)
    {
      if (p.interpolation == GiPrimvarInterpolation::Constant)
      {
        continue;
      }
      uint32_t sizeAndStride = _PrimvarTypeSize(p.type);
      streams.push_back(meshopt_Stream{ (void*) &p.data[0], sizeAndStride, sizeAndStride });
    }

    std::vector<uint32_t> remap(vertexCount);
    uint32_t newVertexCount = meshopt_generateVertexRemapMulti(remap.data(), &faces[0].v_i[0], faceCount * 3,
                                                               vertexCount, streams.data(), streams.size());

    std::vector<GiVertex> newVertices(newVertexCount);
    meshopt_remapVertexBuffer((void*) newVertices.data(), (void*) &vertices[0],
                              vertexCount, sizeof(GiVertex), remap.data());

    std::vector<GiFace> newFaces(faceCount);
    meshopt_remapIndexBuffer((uint32_t*) &newFaces[0], &faces[0].v_i[0], faceCount * 3, remap.data());

    std::vector newPrimvars = primvars;
    for (uint32_t i = 0; i < primvars.size(); i++)
    {
      const auto& o = primvars[i];

      if (o.interpolation == GiPrimvarInterpolation::Constant)
      {
        continue;
      }

      auto& n = newPrimvars[i];
      uint32_t typeSize = _PrimvarTypeSize(o.type);

      n.data.resize(typeSize * vertexCount);
      meshopt_remapVertexBuffer(&n.data[0], &o.data[0], vertexCount, typeSize, remap.data());
    }

    GiMeshData cpuData = {
      .faces = newFaces,
      .primvars = newPrimvars,
      .vertices = newVertices
    };

    return cpuData;
  }
}
