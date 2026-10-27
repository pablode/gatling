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

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfIO.h>
#include <ImfMultiPartInputFile.h>
#include <half.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
  constexpr size_t RGBA_CHANNEL_COUNT = 4;
  constexpr int NO_SPECIAL_INDEX = 10000;

  class _MemStream : public Imf::IStream
  {
  private:
    const char* m_data;
    uint64_t m_size;
    uint64_t m_pos;

  public:
    _MemStream(const char* data, size_t size)
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
      memcpy(c, &m_data[m_pos], n);
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

  uint8_t _FloatToByte(float value)
  {
    value = std::min(std::max(0.0f, value), 1.0f);
    return (uint8_t) std::floor(255.0f * value);
  }

  bool _EqualsIgnoreCase(std::string_view a, std::string_view b)
  {
    if (a.size() != b.size())
    {
      return false;
    }

    for (size_t i = 0; i < a.size(); i++)
    {
      if (std::tolower((unsigned char) a[i]) != std::tolower((unsigned char) b[i]))
      {
        return false;
      }
    }
    return true;
  }

  void _SplitChannelName(const std::string& name, std::string& layer, std::string& suffix)
  {
    const size_t dot = name.rfind('.');

    if (dot == std::string::npos)
    {
      layer.clear();
      suffix = name;
    }
    else
    {
      layer = name.substr(0, dot + 1); // the layer prefix includes the trailing dot
      suffix = name.substr(dot + 1);
    }
  }

  // OpenEXR stores channels in a name-sorted map, so there is no file order we could preserve.
  // Mirror OIIO's channel ordering (see exrinput.cpp).
  int _SpecialIndex(std::string_view suffix, bool xyz)
  {
    static constexpr std::array<std::string_view, 20> GENERAL = {
      "R", "Red", "G", "Green", "B", "Blue", "Y", "real", "imag", "A", "Alpha",
      "AR", "RA", "AG", "GA", "AB", "BA", "Z", "Depth", "Zback"
    };
    static constexpr std::array<std::string_view, 21> XYZ = {
      "R", "Red", "G", "Green", "B", "Blue", "X", "Y", "Z", "real", "imag", "A",
      "Alpha", "AR", "RA", "AG", "GA", "AB", "BA", "Depth", "Zback"
    };

    auto find = [suffix](const auto& names) {
      for (size_t i = 0; i < names.size(); i++)
      {
        if (_EqualsIgnoreCase(suffix, names[i]))
        {
          return (int) i;
        }
      }
      return NO_SPECIAL_INDEX;
    };

    return xyz ? find(XYZ) : find(GENERAL);
  }
}

namespace gtl
{
  namespace
  {
    struct _ExrLayout
    {
      Imath::Box2i dataWindow;
      std::array<std::string, RGBA_CHANNEL_COUNT> rgbaChannelNames; // trailing slots empty
      int alphaChannelIndex = -1;
    };

    _ExrLayout _ReadLayout(const char* data, size_t size)
    {
      _ExrLayout layout;

      _MemStream headerStream(data, size);
      Imf::MultiPartInputFile headerFile(headerStream);
      const Imf::Header& header = headerFile.header(0);
      const Imf::ChannelList& channels = header.channels();

      layout.dataWindow = header.dataWindow();

      struct ChannelDesc
      {
        std::string name;
        std::string layer;
        std::string suffix;
        int specialIndex = NO_SPECIAL_INDEX;
      };

      std::vector<ChannelDesc> channelDescs;
      for (auto it = channels.begin(); it != channels.end(); ++it)
      {
        ChannelDesc desc;
        desc.name = it.name();
        _SplitChannelName(desc.name, desc.layer, desc.suffix);
        channelDescs.push_back(std::move(desc));
      }

      // Layers containing X and Y or Z are treated as normal maps and get a different order.
      for (ChannelDesc& desc : channelDescs)
      {
        bool hasX = false;
        bool hasYOrZ = false;
        for (const ChannelDesc& other : channelDescs)
        {
          if (other.layer != desc.layer)
          {
            continue;
          }
          hasX |= _EqualsIgnoreCase(other.suffix, "X");
          hasYOrZ |= _EqualsIgnoreCase(other.suffix, "Y") ||
                     _EqualsIgnoreCase(other.suffix, "Z");
        }
        desc.specialIndex = _SpecialIndex(desc.suffix, hasX && hasYOrZ);
      }

      std::sort(channelDescs.begin(), channelDescs.end(), [](const ChannelDesc& a, const ChannelDesc& b) {
        if (a.layer != b.layer)
        {
          return a.layer < b.layer;
        }
        if (a.specialIndex != b.specialIndex)
        {
          return a.specialIndex < b.specialIndex;
        }
        return a.suffix < b.suffix;
      });

      for (size_t i = 0; i < layout.rgbaChannelNames.size() && i < channelDescs.size(); i++)
      {
        layout.rgbaChannelNames[i] = channelDescs[i].name;
      }

      for (size_t i = 0; i < channelDescs.size(); i++)
      {
        const std::string& suffix = channelDescs[i].suffix;
        if (_EqualsIgnoreCase(suffix, "A") || _EqualsIgnoreCase(suffix, "Alpha"))
        {
          layout.alphaChannelIndex = (int) i;
          break;
        }
      }

      return layout;
    }

    void _WritePixel(ImgioImage* img, bool keepHdr, long h, long w, float r, float g, float b, float a)
    {
      const uint32_t bytesPerPixel = keepHdr ? 8 : 4;
      const uint64_t offset = (w + h * img->width) * bytesPerPixel;

      if (keepHdr)
      {
        // copy 16 bit floats
        half hr(r), hg(g), hb(b), ha(a);
        memcpy(&img->data[offset + 0], &hr, 2);
        memcpy(&img->data[offset + 2], &hg, 2);
        memcpy(&img->data[offset + 4], &hb, 2);
        memcpy(&img->data[offset + 6], &ha, 2);
      }
      else
      {
        // convert to bytes
        img->data[offset + 0] = _FloatToByte(r);
        img->data[offset + 1] = _FloatToByte(g);
        img->data[offset + 2] = _FloatToByte(b);
        img->data[offset + 3] = _FloatToByte(a);
      }
    }

    void _DecodeRgba(const char* data, size_t size, const _ExrLayout& layout, ImgioImage* img,
                     bool keepHdr, bool greyscaleChannelToRgb)
    {
      _MemStream stream(data, size);
      Imf::InputFile file(stream);

      const long width = (long) img->width;
      const long height = (long) img->height;
      const Imath::Box2i& dw = layout.dataWindow;
      size_t pixelCount = size_t(width) * size_t(height);

      std::vector<half> pixels(pixelCount * RGBA_CHANNEL_COUNT, half(0.0f));

      // Channels are assigned contiguously from R onwards
      size_t fileChannelCount = 0;
      while (fileChannelCount < RGBA_CHANNEL_COUNT &&
             !layout.rgbaChannelNames[fileChannelCount].empty())
      {
        fileChannelCount++;
      }

      const size_t xStride = RGBA_CHANNEL_COUNT * sizeof(half);
      const size_t yStride = xStride * (size_t) width;

      if (fileChannelCount > 0)
      {
        Imf::FrameBuffer frameBuffer;
        for (size_t slot = 0; slot < fileChannelCount; slot++)
        {
          const std::string& name = layout.rgbaChannelNames[slot];

          char* base = (char*) pixels.data() + slot * sizeof(half);
          base -= (ptrdiff_t) dw.min.x * (ptrdiff_t) xStride;
          base -= (ptrdiff_t) dw.min.y * (ptrdiff_t) yStride;

          frameBuffer.insert(name.c_str(), Imf::Slice(Imf::HALF, base, xStride, yStride));
        }
        file.setFrameBuffer(frameBuffer);
        file.readPixels(dw.min.y, dw.max.y);
      }

      // Mirror OIIO's gray_to_rgb handling (TextureSystem::fill_gray_channels): a single
      // channel is replicated to RGB, and a two-channel image whose second channel is alpha
      // is turned into RRRA. Missing channels stay zero, and alpha defaults to one.
      if (fileChannelCount < RGBA_CHANNEL_COUNT)
      {
        for (size_t i = 0; i < pixelCount; i++)
        {
          half* pixel = pixels.data() + i * RGBA_CHANNEL_COUNT;

          if (greyscaleChannelToRgb)
          {
            if (fileChannelCount == 1)
            {
              pixel[1] = pixel[2] = pixel[0];
            }
            else if (fileChannelCount == 2 && layout.alphaChannelIndex == 1)
            {
              pixel[3] = pixel[1];
              pixel[1] = pixel[2] = pixel[0];
              continue;
            }
          }

          pixel[3] = half(1.0f);
        }
      }

      for (long h = 0; h < height; h++)
      for (long w = 0; w < width; w++)
      {
        const half* pixel = pixels.data() + (((height - 1 - h) * width + w) * RGBA_CHANNEL_COUNT);
        _WritePixel(img, keepHdr, h, w, (float) pixel[0], (float) pixel[1], (float) pixel[2], (float) pixel[3]);
      }
    }

    void _DecodeExr(const char* data, size_t size, ImgioImage* img, bool keepHdr,
                    bool greyscaleChannelToRgb)
    {
      const _ExrLayout layout = _ReadLayout(data, size);
      const Imath::Box2i& dw = layout.dataWindow;

      const uint32_t bytesPerPixel = keepHdr ? 8 : 4;

      img->width = (dw.max.x - dw.min.x + 1);
      img->height = (dw.max.y - dw.min.y + 1);
      img->size = size_t(img->width) * size_t(img->height) * bytesPerPixel;
      img->data.resize(img->size);
      img->format = keepHdr ? ImgioFormat::RGBA16_FLOAT : ImgioFormat::RGBA8_UNORM;

      _DecodeRgba(data, size, layout, img, keepHdr, greyscaleChannelToRgb);
    }
  }

  ImgioError ImgioExrDecoder::decode(size_t size, const void* data, ImgioImage* img,
                                     bool keepHdr, bool greyscaleChannelToRgb)
  {
    // Do the signature check manually because we can't detect a mismatch based on the exception-based API.
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
      _DecodeExr(static_cast<const char*>(data), size, img, keepHdr, greyscaleChannelToRgb);
    }
    catch (const std::exception&)
    {
      *img = {}; // free memory
      return ImgioError::Decode;
    }

    return ImgioError::None;
  }
}
