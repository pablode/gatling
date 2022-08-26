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

#include "exr.h"

#include "img.h"
#include "error_codes.h"

#include <ImfRgbaFile.h>
#include <ImfIO.h>
#include <ImfArray.h>
#include <ImfRgba.h>

#include <algorithm>

class MemStream : public Imf::IStream
{
private:
  uint64_t m_pos;
  uint64_t m_size;
  char* m_data;

public:
  MemStream(char* data, size_t size)
    : Imf::IStream("")
    , m_data(data)
    , m_size(size)
    , m_pos(0)
  {
  }

  bool isMemoryMapped() const override
  {
    return true;
  }

  bool read(char c[], int n) override
  {
    if (m_pos + n > m_size)
    {
      return false;
    }
    memcpy(c, (void*) &m_data[m_pos], n);
    m_pos += n;
    return true;
  }

  char* readMemoryMapped(int n) override
  {
    if (m_pos + n > m_size)
    {
      throw std::invalid_argument("Trying to read out of bounds");
    }
    char* ptr = &m_data[m_pos];
    m_pos += n;
    return ptr;
  }

  uint64_t tellg() override
  {
    return m_pos;
  }

  void seekg(uint64_t pos) override
  {
    m_pos = pos;
  }
};

uint8_t _FloatToByte(float value)
{
  int result = std::floor(255 * value + 0.499999f);
  return std::min(255, std::max(0, result));
}

int imgio_exr_decode(size_t size,
                     void* mem,
                     imgio_img* img)
{
  // Do the signature check manually because we can't detect
  // a mismatch based on the exception-based API.
  if (size < 4)
  {
    return IMGIO_ERR_IO;
  }

  uint8_t signature[] = { 0x76, 0x2F, 0x31, 0x01 };
  if (memcmp(mem, signature, 4))
  {
    return IMGIO_ERR_UNSUPPORTED_ENCODING;
  }

  img->data = nullptr;

  try
  {
    MemStream stream((char*) mem, size);

    Imf::RgbaInputFile file(stream);

    const Imath::Box2i& window = file.dataWindow();
    img->width = (window.max.x - window.min.x + 1);
    img->height = (window.max.y - window.min.y + 1);
    img->size = img->width * img->height * 4;
    img->data = (uint8_t*) malloc(img->size);

    Imf::Array2D<Imf::Rgba> tmpPixels(img->width, img->height); // values are 16-bit floats
    file.setFrameBuffer(&tmpPixels[0][0] - window.min.x - window.min.y * img->width, 1, img->width);
    file.readPixels(window.min.y, window.max.y);

    for (long h = 0; h < tmpPixels.height(); h++)
    {
      for (long w = 0; w < tmpPixels.width(); w++)
      {
        const Imf::Rgba& value = tmpPixels[w][h];

        uint64_t offset = (w + h * img->width) * 4;
        img->data[offset + 0] = _FloatToByte(value.r);
        img->data[offset + 1] = _FloatToByte(value.g);
        img->data[offset + 2] = _FloatToByte(value.b);
        img->data[offset + 3] = _FloatToByte(value.a);
      }
    }
  }
  catch (std::exception& ex)
  {
    free(img->data);
    return IMGIO_ERR_DECODE;
  }

  return IMGIO_OK;
}
