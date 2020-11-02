#ifndef GATLING_IMG_H
#define GATLING_IMG_H

#include <stdbool.h>
#include <stdint.h>

bool gatling_img_write(
  const uint8_t* data,
  uint32_t width,
  uint32_t height,
  const char* path
);

#endif
