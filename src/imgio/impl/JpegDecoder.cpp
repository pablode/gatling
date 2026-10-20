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

#include "JpegDecoder.h"
#include "ErrorCodes.h"
#include "Image.h"

#include <stdlib.h>
#include <turbojpeg.h>

namespace gtl
{
  ImgioError ImgioJpegDecoder::decode(size_t size, const void* data, ImgioImage* img)
  {
    tjhandle instance = tjInitDecompress();
    if (!instance)
    {
      return ImgioError::Unknown;
    }

    int subsamp;
    int colorspace;
    int width;
    int height;
    if (tjDecompressHeader3(instance, (const unsigned char*) data, (unsigned long) size,
                            &width, &height,
                            &subsamp, &colorspace) < 0)
    {
      tjDestroy(instance);
      return ImgioError::UnsupportedEncoding;
    }

    img->width = (uint32_t) width;
    img->height = (uint32_t) height;

    int pixelFormat = TJPF_RGBA;
    img->format = ImgioFormat::RGBA8_UNORM;
    img->size = size_t(img->width) * img->height * tjPixelSize[pixelFormat];
    img->data.resize(img->size);

    int result = tjDecompress2(instance, (const unsigned char*) data, (unsigned long) size,
                               (unsigned char*) &img->data[0], (int) img->width, 0,
                               (int) img->height, pixelFormat, TJFLAG_ACCURATEDCT | TJFLAG_BOTTOMUP);
    tjDestroy(instance);

    if (result < 0)
    {
      *img = {}; // free memory
      return ImgioError::Decode;
    }

    return ImgioError::None;
  }
}
