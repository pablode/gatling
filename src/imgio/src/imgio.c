#include "imgio.h"

#include "mmap.h"
#include "png.h"
#include "jpeg.h"

#include <stdlib.h>

int imgio_load_img(const char* file_path,
                   struct imgio_img* img)
{
  struct imgio_file* file;
  if (!imgio_file_open(file_path,
                       IMGIO_FILE_USAGE_READ,
                       &file))
  {
    return IMGIO_ERR_FILE_NOT_FOUND;
  }

  size_t size = imgio_file_size(file);
  void* data = imgio_mmap(file, 0, size);
  if (!data)
  {
    imgio_file_close(file);
    return IMGIO_ERR_IO;
  }

  int r = imgio_png_decode(size, data, img);

  if (r == IMGIO_ERR_UNSUPPORTED_ENCODING)
  {
    r = imgio_jpeg_decode(size, data, img);
  }

  imgio_munmap(file, data);
  imgio_file_close(file);

  return r;
}

void imgio_free_img(struct imgio_img* img)
{
  free(img->data);
}
