//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#include "TiffDecoder.h"
#include "ErrorCodes.h"
#include "Image.h"

#include <tiffio.h>
#include <tiffio.hxx>

#include <sstream>

namespace gtl
{
  ImgioError ImgioTiffDecoder::decode(size_t size, const void* data, ImgioImage* img)
  {
    const auto cData = (char*) data;
    std::istringstream stream(std::string(cData, cData + size)); // FIXME: don't copy; use custom istream

    TIFF* tiff = TIFFStreamOpen("MemTIFF", &stream);
    if (!tiff)
    {
      return ImgioError::UnsupportedEncoding;
    }

    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &img->width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &img->height);

    img->size = img->width * img->height * 4;
    img->data.resize(img->size);

    int result = TIFFReadRGBAImageOriented(tiff, img->width, img->height, (uint32_t*) &img->data[0], ORIENTATION_TOPLEFT, 1);

    TIFFClose(tiff);

    return result ? ImgioError::None : ImgioError::Decode;
  }
}
