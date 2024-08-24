//
// Copyright (C) 2023 Pablo Delgado Krämer
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
#include <stdbool.h>
#include <stdlib.h>
#include <vector>

namespace gtl
{
  class GbHandleStore
  {
  public:
    GbHandleStore();

    uint64_t allocateHandle();

    bool isHandleValid(uint64_t handle) const;

    void freeHandle(uint64_t handle);

  private:
    uint32_t m_maxIndex = 0;
    std::vector<uint32_t> m_versions;
    std::vector<uint32_t> m_freeList;
  };
}
