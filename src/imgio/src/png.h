#ifndef IMGIO_PNG_H
#define IMGIO_PNG_H

#include <stddef.h>

#include "img.h"

int imgio_png_decode(size_t size,
                     void* data,
                     struct imgio_img* img);

#endif
