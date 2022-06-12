/*
 * This file is part of gatling.
 *
 * Copyright (C) 2019-2022 Pablo Delgado Krämer
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

#include "jpeg.h"

#include "error_codes.h"

#include <stdlib.h>
#include <turbojpeg.h>

int imgio_jpeg_decode(size_t size,
                      void* mem,
                      struct imgio_img* img)
{
  tjhandle instance = tjInitDecompress();
  if (!instance)
  {
    return IMGIO_ERR_UNKNOWN;
  }

  int subsamp;
  int colorspace;
  if (tjDecompressHeader3(instance, mem, size,
                          (int*) &img->width, (int*) &img->height,
                          &subsamp, &colorspace) < 0)
  {
    tjDestroy(instance);
    return IMGIO_ERR_UNSUPPORTED_ENCODING;
  }

  int pixelFormat = TJPF_RGBA;
  img->size = img->width * img->height * tjPixelSize[pixelFormat];
  img->data = malloc(img->size);

  int result = tjDecompress2(instance, mem, size, (unsigned char*) img->data,
                             (int) img->width, 0, (int) img->height,
                             pixelFormat, TJFLAG_ACCURATEDCT);
  tjDestroy(instance);

  if (result < 0)
  {
    free(img->data);
    return IMGIO_ERR_DECODE;
  }

  return IMGIO_OK;
}
