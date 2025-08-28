//
// Copyright (C) 2025 Pablo Delgado Krämer
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

#include "DdsDecoder.h"
#include "ErrorCodes.h"
#include "Image.h"

#include <ddspp.h>
#include <sstream>

namespace
{
  using namespace gtl;

  ImgioFormat _TranslateFormat(ddspp::DXGIFormat format)
  {
    switch (format)
    {
    case ddspp::R8G8B8A8_UINT: return ImgioFormat::RGBA8_UNORM;
    case ddspp::R32_FLOAT: return ImgioFormat::R32_FLOAT;
    case ddspp::BC1_UNORM: return ImgioFormat::BC1_UNORM;
    case ddspp::BC1_UNORM_SRGB: return ImgioFormat::BC1_UNORM_SRGB;
    case ddspp::BC2_UNORM: return ImgioFormat::BC2_UNORM;
    case ddspp::BC2_UNORM_SRGB: return ImgioFormat::BC2_UNORM_SRGB;
    case ddspp::BC3_UNORM: return ImgioFormat::BC3_UNORM;
    case ddspp::BC3_UNORM_SRGB: return ImgioFormat::BC3_UNORM_SRGB;
    case ddspp::BC4_UNORM: return ImgioFormat::BC4_UNORM;
    case ddspp::BC4_SNORM: return ImgioFormat::BC4_SNORM;
    case ddspp::BC5_UNORM: return ImgioFormat::BC5_UNORM;
    case ddspp::BC5_SNORM: return ImgioFormat::BC5_SNORM;
    case ddspp::BC7_UNORM: return ImgioFormat::BC7_UNORM;
    case ddspp::BC7_UNORM_SRGB: return ImgioFormat::BC7_UNORM_SRGB;
    default: return ImgioFormat::UNSUPPORTED;
    };
  }
}

namespace gtl
{
  ImgioError ImgioDdsDecoder::decode(size_t size, const void* data, ImgioImage* img)
  {
    ddspp::Descriptor desc;
    ddspp::Result decodeResult = ddspp::decode_header((const unsigned char*) data, desc);

    if (decodeResult != ddspp::Result::Success)
    {
      return ImgioError::UnsupportedEncoding;
    }

    if (desc.type != ddspp::TextureType::Texture2D)
    {
      return ImgioError::UnsupportedFeature;
    }

    const uint8_t* mip0Data = &((const uint8_t*) data)[desc.headerSize];

    size_t mip0Size;
    if (desc.compressed)
    {
      mip0Size = desc.width * desc.height * desc.bitsPerPixelOrBlock / (8 * desc.blockWidth * desc.blockHeight);
    }
    else
    {
      mip0Size = desc.width * desc.bitsPerPixelOrBlock / 8;
    }

    img->format = _TranslateFormat(desc.format);
    img->width = desc.width;
    img->height = desc.height;
    img->size = mip0Size;
    img->data = std::vector<uint8_t>(mip0Data, &mip0Data[mip0Size]);

    return ImgioError::None;
  }
}
