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

#include "mmap.h"
#include "png.h"
#include "jpeg.h"

#include <stdlib.h>

int imgio_load_img(const char* file_path,
                   imgio_img* img)
{
  imgio_file* file;
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

void imgio_free_img(imgio_img* img)
{
  free(img->data);
}
