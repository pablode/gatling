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

#include "png.h"

#include "img.h"
#include "error_codes.h"

#include <stdlib.h>
#include <spng.h>

int imgio_png_decode(size_t size,
                     void* mem,
                     imgio_img* img)
{
  int err;

  spng_ctx* ctx = spng_ctx_new(0);

  err = spng_set_png_buffer(ctx, mem, size);
  if (err != SPNG_OK)
  {
    goto buffer_fail;
  }

  err = spng_decoded_image_size(ctx, SPNG_FMT_RGBA8, &img->size);
  if (err != SPNG_OK)
  {
    goto decode_size_fail;
  }

  img->data = (uint8_t*) malloc(img->size);

  err = spng_decode_image(ctx, img->data, img->size, SPNG_FMT_RGBA8, 0);
  if (err != SPNG_OK)
  {
    goto decode_fail;
  }

  spng_ihdr ihdr;
  err = spng_get_ihdr(ctx, &ihdr);
  if (err != SPNG_OK)
  {
    goto ihdr_fail;
  }

  img->width = ihdr.width;
  img->height = ihdr.height;

  spng_ctx_free(ctx);

  return IMGIO_OK;

ihdr_fail:
decode_fail:
  free(img->data);

decode_size_fail:
buffer_fail:
  spng_ctx_free(ctx);

  if (err == SPNG_ESIGNATURE)
  {
    return IMGIO_ERR_UNSUPPORTED_ENCODING;
  }
  else if (err == SPNG_IO_ERROR || err == SPNG_IO_EOF)
  {
    return IMGIO_ERR_IO;
  }
  else
  {
    return IMGIO_ERR_DECODE;
  }
}
