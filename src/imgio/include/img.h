#ifndef IMGIO_IMG_H
#define IMGIO_IMG_H

#include <stddef.h>
#include <stdint.h>

struct imgio_img
{
  void* data;
  size_t size;
  uint32_t width;
  uint32_t height;
};

#endif
