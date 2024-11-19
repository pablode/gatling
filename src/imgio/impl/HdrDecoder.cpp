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

#include "HdrDecoder.h"
#include "ErrorCodes.h"
#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STBI_ONLY_HDR
#include <stb_image.h>

#include <algorithm>
#include <float.h>

namespace gtl
{
  // FIXME: don't downcast to uint8_t; store as RGBA16F / E5B9G9R9_UFLOAT_PACK32 / ASTC_HDR
  ImgioError ImgioHdrDecoder::decode(size_t size, const void* data, ImgioImage* img)
  {
    if (!stbi_is_hdr_from_memory((const stbi_uc*) data, (int) size))
    {
      return ImgioError::UnsupportedEncoding;
    }

    stbi_set_flip_vertically_on_load(1);

    int num_components;
    float* hdrData = stbi_loadf_from_memory((const stbi_uc*) data, (int) size, (int*) &img->width, (int*) &img->height, &num_components, 4);
    if (!hdrData)
    {
      return ImgioError::Decode;
    }

    img->size = img->width * img->height * 4;
    img->data.resize(img->size);

    for (uint32_t i = 0; i < img->width; i++)
    {
      for (uint32_t j = 0; j < img->height; j++)
      {
        for (uint32_t k = 0; k < 4; k++)
        {
          uint32_t cIndex = (i + j * img->width) * 4 + k;
          img->data[cIndex] = uint8_t(std::min(int(hdrData[cIndex] * 255.0f), 255));
        }
      }
    }

    delete hdrData;
    return ImgioError::None;
  }
}
