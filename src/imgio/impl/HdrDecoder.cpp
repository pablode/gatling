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
#define STB_IMAGE_STATIC
#define STBI_FAILURE_USERMSG
#define STBI_ONLY_HDR
#include <stb_image.h>

#include <algorithm>
#include <float.h>
#include <half.h>
#include <limits.h>

namespace gtl
{
  ImgioError ImgioHdrDecoder::decode(size_t size, const void* data, ImgioImage* img, bool keepHdr)
  {
    if (!stbi_is_hdr_from_memory((const stbi_uc*) data, (int) size))
    {
      return ImgioError::UnsupportedEncoding;
    }

    stbi_set_flip_vertically_on_load(1);

    int width;
    int height;
    float* hdrData = stbi_loadf_from_memory((const stbi_uc*) data, (int) size, &width, &height, nullptr, 4);

    if (!hdrData)
    {
      return ImgioError::Decode;
    }

    img->width = (uint32_t) width;
    img->height = (uint32_t) height;

    uint32_t bytesPerPixel = keepHdr ? 8 : 4;

    img->size = size_t(img->width) * img->height * bytesPerPixel;
    img->data.resize(img->size, 0);
    img->format = keepHdr ? ImgioFormat::RGBA16_FLOAT : ImgioFormat::RGBA8_UNORM;

    const float* floatDataIn = (const float*) hdrData;
    uint8_t* byteDataOut = &img->data[0];
    half* halfDataOut = (half*) &img->data[0];

    for (uint32_t i = 0; i < img->width; i++)
    for (uint32_t j = 0; j < img->height; j++)
    {
      uint32_t idx = (i + j * img->width) * 4;

      for (uint32_t k = 0; k < 4; k++)
      {
        if (keepHdr)
        {
          halfDataOut[idx + k] = half(floatDataIn[idx + k]);
        }
        else
        {
          float value = std::min(std::max(0.0f, floatDataIn[idx + k]), 1.0f);
          byteDataOut[idx + k] = uint8_t(value * 255.0f);
        }
      }
    }

    stbi_image_free(hdrData);
    return ImgioError::None;
  }
}
