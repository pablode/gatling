/*
 * This file is part of gatling.
 *
 * Copyright (C) 2023 Pablo Delgado Krämer
 *
 * gatling is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "hdr.h"

#include "img.h"
#include "error_codes.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STBI_ONLY_HDR
#include <stb_image.h>

#include <algorithm>
#include <float.h>

// FIXME: don't downcast to uint8_t; store as RGBA16F / R11FB10F / ASTC_HDR
int imgio_hdr_decode(size_t size, const void* data, imgio_img* img)
{
  if (!stbi_is_hdr_from_memory((const stbi_uc*) data, size))
  {
    return IMGIO_ERR_UNSUPPORTED_ENCODING;
  }

  int num_components;
  float* hdrData = stbi_loadf_from_memory((const stbi_uc*)data, size, (int*)&img->width, (int*)&img->height, &num_components, 4);
  if (!hdrData)
  {
    return IMGIO_ERR_DECODE;
  }

  img->size = img->width * img->height * 4;
  img->data = (uint8_t*) malloc(img->size);

  for (uint32_t i = 0; i < img->width; i++)
  {
    for (uint32_t j = 0; j < img->height; j++)
    {
      for (uint32_t k = 0; k < 4; k++)
      {
        uint32_t cIndex = (i + j * img->width) * 4 + k;
        img->data[cIndex] = uint8_t(std::min(int(hdrData[cIndex] * 255.0), 255));
      }
    }
  }

  delete hdrData;
  return IMGIO_OK;
}
