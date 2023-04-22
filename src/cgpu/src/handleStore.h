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

#include <smallVector.h>

namespace gtl
{
  class GbHandleStore
  {
  public:
    uint64_t allocateHandle();

    bool isHandleValid(uint64_t handle) const;

    void freeHandle(uint64_t handle);

  private:
    uint32_t m_maxIndex = 0;
    GbSmallVector<uint32_t, 1024> m_versions;
    GbSmallVector<uint32_t, 1024> m_freeList;
  };
}
