#ifndef IMGIO_JPEG_H
#define IMGIO_JPEG_H

#include <stddef.h>

#include "img.h"

int imgio_jpeg_decode(size_t size,
                      void* data,
                      struct imgio_img* img);

#endif
