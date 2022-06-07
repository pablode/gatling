//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
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

#include "plugin.h"

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/iimage_plugin.h>

#include <vector>
#include <cassert>
#include <string>

namespace detail
{
  class MdlImageTile : public mi::base::Interface_implement<mi::neuraylib::ITile>
  {
  public:
    MdlImageTile()
    {
      // Magenta is the default color.
      m_data[0] = 255;
      m_data[1] = 0;
      m_data[2] = 255;
      m_data[3] = 0;
    }

    void get_pixel(mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const override
    {
      assert(false);
    }

    void set_pixel(mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats) override
    {
      assert(false);
    }

    const char* get_type() const override
    {
      return "Rgba";
    }

    mi::Uint32 get_resolution_x() const override
    {
      return 1;
    }

    mi::Uint32 get_resolution_y() const override
    {
      return 1;
    }

    const void* get_data() const override
    {
      return m_data;
    }

    void* get_data() override
    {
      return m_data;
    }

  private:
    uint8_t m_data[4];
  };

  class MdlImageFile : public mi::base::Interface_implement<mi::neuraylib::IImage_file>
  {
  public:
    MdlImageFile()
      : m_tile(new MdlImageTile())
    {
    }

    const char* get_type() const override
    {
      return "Rgba";
    }

    mi::Uint32 get_resolution_x(mi::Uint32 level = 0) const override
    {
      assert(level == 0);
      return m_tile->get_resolution_x();
    }

    mi::Uint32 get_resolution_y(mi::Uint32 level = 0) const override
    {
      assert(level == 0);
      return m_tile->get_resolution_y();
    }

    mi::Uint32 get_layers_size(mi::Uint32 level = 0) const override
    {
      return 1;
    }

    mi::Uint32 get_miplevels() const override
    {
      return 0;
    }

    bool get_is_cubemap() const override
    {
      return false;
    }

    mi::Float32 get_gamma() const override
    {
      return 1.0f;
    }

    mi::neuraylib::ITile* read(mi::Uint32 z, mi::Uint32 level = 0) const override
    {
      assert(m_tile);
      return m_tile.get();
    }

    bool write(const mi::neuraylib::ITile* tile, mi::Uint32 z, mi::Uint32 level = 0) override
    {
      assert(false);
      return false;
    }

  private:
    mi::base::Handle<MdlImageTile> m_tile;
  };

  class ImagePlugin : public mi::neuraylib::IImage_plugin
  {
  public:
    bool init(mi::neuraylib::IPlugin_api* plugin_api) override
    {
      return true;
    }

    bool exit(mi::neuraylib::IPlugin_api* plugin_api) override
    {
      return true;
    }

    const char* get_supported_type(mi::Uint32 index) const override
    {
      return (index == 0) ? "Rgba" : nullptr;
    }

    mi::neuraylib::Impexp_priority get_priority() const override
    {
      return mi::neuraylib::IMPEXP_PRIORITY_OVERRIDE;
    }

    mi::neuraylib::IImage_file* open_for_writing(mi::neuraylib::IWriter* writer,
                                                 const char* pixel_type,
                                                 mi::Uint32 resolution_x,
                                                 mi::Uint32 resolution_y,
                                                 mi::Uint32 nr_of_layers,
                                                 mi::Uint32 miplevels,
                                                 bool is_cubemap,
                                                 mi::Float32 gamma,
                                                 mi::Uint32 quality) const override
    {
      assert(false);
      return nullptr;
    }

    mi::neuraylib::IImage_file* open_for_reading(mi::neuraylib::IReader* reader) const override
    {
      return new MdlImageFile();
    }

  public:
    const char* get_type() const override
    {
      return MI_NEURAY_IMAGE_PLUGIN_TYPE;
    }

    void release() override
    {
    }
  };

  class JpegImagePlugin : public ImagePlugin
  {
  public:
    const char* get_name() const override
    {
      return "gatling_jpg_loader";
    }

    const char* get_file_extension(mi::Uint32 index) const override
    {
      switch (index)
      {
      case 0: return "jpg";
      case 1: return "jpeg";
      default: return nullptr;
      }
    }

    bool test(const mi::Uint8* buffer /* of size 512b */, mi::Uint32 file_size) const override
    {
      return (file_size >= 3) && (buffer[0] == 0xFF && buffer[1] == 0xD8 && buffer[2] == 0xFF);
    }
  };

  class PngImagePlugin : public ImagePlugin
  {
  public:
    const char* get_name() const override
    {
      return "gatling_png_loader";
    }

    const char* get_file_extension(mi::Uint32 index) const override
    {
      switch (index)
      {
      case 0: return "png";
      default: return nullptr;
      }
    }

    bool test(const mi::Uint8* buffer /* 512b */, mi::Uint32 file_size) const override
    {
      return (file_size >= 8) && (buffer[0] == 0x89 && buffer[1] == 0x50 && buffer[2] == 0x4E &&
                                  buffer[3] == 0x47 && buffer[4] == 0x0D && buffer[5] == 0x0A &&
                                  buffer[6] == 0x1A && buffer[7] == 0x0A);
    }
  };
}

extern "C" MI_DLL_EXPORT mi::base::Plugin* mi_plugin_factory(mi::Sint32 index, void* context)
{
  switch (index)
  {
  case 0: return new detail::JpegImagePlugin();
  case 1: return new detail::PngImagePlugin();
  default: return nullptr;
  }
}
