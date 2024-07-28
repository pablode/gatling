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

#include "ExrDecoder.h"
#include "ErrorCodes.h"
#include "Image.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <ImfRgbaFile.h>
#include <ImfIO.h>
#include <ImfArray.h>
#include <ImfRgba.h>
#pragma clang diagnostic pop

#include <algorithm>
#include <assert.h>

namespace
{
  class _MemStream : public Imf::IStream
  {
  private:
    char* m_data;
    uint64_t m_size;
    uint64_t m_pos;

  public:
    _MemStream(char* data, size_t size)
      : Imf::IStream("")
      , m_data(data)
      , m_size(size)
      , m_pos(0)
    {
    }

    bool read(char c[], int n) override
    {
      assert(m_data);
      if (m_pos + n > m_size)
      {
        return false;
      }
      memcpy(c, (void*)&m_data[m_pos], n);
      m_pos += n;
      return true;
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

  uint8_t _floatToByte(float value)
  {
    int result = (int)std::floor(255 * value + 0.499999f);
    return std::min(255, std::max(0, result));
  }
}

namespace gtl
{
  ImgioError ImgioExrDecoder::decode(size_t size, const void* data, ImgioImage* img)
  {
    // Do the signature check manually because we can't detect
    // a mismatch based on the exception-based API.
    if (size < 4)
    {
      return ImgioError::CorruptData;
    }

    uint8_t signature[] = { 0x76, 0x2F, 0x31, 0x01 };
    if (memcmp(data, signature, 4))
    {
      return ImgioError::UnsupportedEncoding;
    }

    try
    {
      _MemStream stream((char*)data, size);

      Imf::RgbaInputFile file(stream);

      const Imath::Box2i& dw = file.dataWindow();
      img->width = (dw.max.x - dw.min.x + 1);
      img->height = (dw.max.y - dw.min.y + 1);
      img->size = img->width * img->height * 4;
      img->data.resize(img->size);

      Imf::Array2D<Imf::Rgba> tmpPixels(img->height, img->width); // values are 16-bit floats
      file.setFrameBuffer(&tmpPixels[0][0] - dw.min.x - dw.min.y * img->width, 1, img->width);
      file.readPixels(dw.min.y, dw.max.y);

      for (long h = 0; h < tmpPixels.height(); h++)
      {
        for (long w = 0; w < tmpPixels.width(); w++)
        {
          const Imf::Rgba& value = tmpPixels[h][w];

          uint64_t offset = (w + h * img->width) * 4;
          img->data[offset + 0] = _floatToByte(value.r);
          img->data[offset + 1] = _floatToByte(value.g);
          img->data[offset + 2] = _floatToByte(value.b);
          img->data[offset + 3] = _floatToByte(value.a);
        }
      }
    }
    catch (std::exception&)
    {
      *img = {}; // free memory
      return ImgioError::Decode;
    }

    return ImgioError::None;
  }
}
