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

  img->size = img->width * img->height * 4;
  img->data = malloc(img->size);

  int result = tjDecompress2(instance, mem, size, (unsigned char*) img->data,
                             (int) img->width, 0, (int) img->height,
                             TJPF_RGBA, TJFLAG_ACCURATEDCT);
  tjDestroy(instance);

  if (result < 0)
  {
    free(img->data);
    return IMGIO_ERR_DECODE;
  }

  return IMGIO_OK;
}
