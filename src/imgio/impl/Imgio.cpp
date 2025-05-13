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

#include "Imgio.h"
#include "PngDecoder.h"
#include "JpegDecoder.h"
#include "ExrDecoder.h"
#include "HdrDecoder.h"
#include "TiffDecoder.h"
#include "TgaDecoder.h"

#include <stdlib.h>

namespace gtl
{
  ImgioError ImgioLoadImage(const void* data, size_t size, ImgioImage* img)
  {
    ImgioError r = ImgioPngDecoder::decode(size, data, img);

    if (r == ImgioError::UnsupportedEncoding)
    {
      r = ImgioJpegDecoder::decode(size, data, img);
    }

    if (r == ImgioError::UnsupportedEncoding)
    {
      r = ImgioExrDecoder::decode(size, data, img);
    }

    if (r == ImgioError::UnsupportedEncoding)
    {
      r = ImgioHdrDecoder::decode(size, data, img);
    }

    if (r == ImgioError::UnsupportedEncoding)
    {
      r = ImgioTiffDecoder::decode(size, data, img);
    }

    if (r == ImgioError::UnsupportedEncoding)
    {
      r = ImgioTgaDecoder::decode(size, data, img);
    }

    return r;
  }
}
