#include "png.h"

#include "error_codes.h"

#include <stdlib.h>
#include <spng.h>

int imgio_png_decode(size_t size,
                     void* mem,
                     struct imgio_img* img)
{
  enum spng_errno err;

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

  img->data = malloc(img->size);

  err = spng_decode_image(ctx, img->data, img->size, SPNG_FMT_RGBA8, 0);
  if (err != SPNG_OK)
  {
    goto decode_fail;
  }

  struct spng_ihdr ihdr;
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
