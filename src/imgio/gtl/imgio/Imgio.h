//
// Copyright (C) 2019 Pablo Delgado Krämer
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

#include "Image.h"
#include "ErrorCodes.h"

#include <gtl/gb/Enum.h>

namespace gtl
{
  enum class ImgioLoadFlags
  {
    None = 0,
    KeepHdr
  };
  GB_DECLARE_ENUM_BITOPS(ImgioLoadFlags);

  ImgioError ImgioLoadImage(const void* data,
                            size_t size,
                            ImgioImage* img,
                            ImgioLoadFlags flags = ImgioLoadFlags::None);
}
