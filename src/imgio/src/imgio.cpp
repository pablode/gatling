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

#include "imgio.h"

#include "png.h"
#include "jpeg.h"
#include "exr.h"

#include <stdlib.h>

int imgio_load_img(const void* data, size_t size, imgio_img* img)
{
  int r = imgio_png_decode(size, data, img);

  if (r == IMGIO_ERR_UNSUPPORTED_ENCODING)
  {
    r = imgio_jpeg_decode(size, data, img);
  }

  if (r == IMGIO_ERR_UNSUPPORTED_ENCODING)
  {
    r = imgio_exr_decode(size, data, img);
  }

  return r;
}

void imgio_free_img(imgio_img* img)
{
  free(img->data);
}
