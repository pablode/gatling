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

#include "TgaDecoder.h"
#include "ErrorCodes.h"
#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_FAILURE_USERMSG
#define STBI_ONLY_TGA
#include <stb_image.h>

#include <string.h>
#include <limits.h>

namespace gtl
{
  ImgioError ImgioTgaDecoder::decode(size_t size, const void* data, ImgioImage* img)
  {
    stbi_set_flip_vertically_on_load(1);

    int width;
    int height;
    int num_components;
    uint8_t* rgbaData = stbi_load_from_memory((const stbi_uc*) data, (int) size, &width, &height, &num_components, 4);
    if (!rgbaData)
    {
      const char* reason = stbi_failure_reason();
      if (reason && strcmp(reason, "Image not of any known type, or corrupt") == 0)
      {
        return ImgioError::UnsupportedEncoding;
      }
      return ImgioError::Decode;
    }

    img->width = (uint32_t) width;
    img->height = (uint32_t) height;
    img->format = ImgioFormat::RGBA8_UNORM;
    img->size = size_t(img->width) * img->height * 4;
    img->data.resize(img->size);
    memcpy(&img->data[0], rgbaData, img->size);

    stbi_image_free(rgbaData);
    return ImgioError::None;
  }
}
