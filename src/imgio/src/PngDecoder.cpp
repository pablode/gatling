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

#include "PngDecoder.h"
#include "ErrorCodes.h"
#include "Image.h"

#include <stdlib.h>
#include <spng.h>

namespace gtl
{
  ImgioError ImgioPngDecoder::decode(size_t size, const void* data, ImgioImage* img)
  {
    int err;

    spng_ctx* ctx = spng_ctx_new(0);

    err = spng_set_png_buffer(ctx, data, size);
    if (err != SPNG_OK)
    {
      goto fail;
    }

    err = spng_decoded_image_size(ctx, SPNG_FMT_RGBA8, &img->size);
    if (err != SPNG_OK)
    {
      goto fail;
    }

    img->data.resize(img->size);

    err = spng_decode_image(ctx, &img->data[0], img->size, SPNG_FMT_RGBA8, 0);
    if (err != SPNG_OK)
    {
      goto fail;
    }

    spng_ihdr ihdr;
    err = spng_get_ihdr(ctx, &ihdr);
    if (err != SPNG_OK)
    {
      goto fail;
    }

    img->width = ihdr.width;
    img->height = ihdr.height;

    spng_ctx_free(ctx);

    return ImgioError::None;

  fail:
    *img = {}; // free memory

    spng_ctx_free(ctx);

    if (err == SPNG_ESIGNATURE)
    {
      return ImgioError::UnsupportedEncoding;
    }
    else if (err == SPNG_IO_ERROR || err == SPNG_IO_EOF)
    {
      return ImgioError::CorruptData;
    }
    else
    {
      return ImgioError::Decode;
    }
  }
}
