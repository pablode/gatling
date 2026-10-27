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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <filesystem>
#include <array>
#include <half.h>

#include "Imgio.h"

namespace fs = std::filesystem;
using namespace gtl;

bool _ReadFile(const fs::path& filePath, std::vector<uint8_t>& data)
{
  std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
  if (!file.is_open())
  {
    return false;
  }
  file.seekg(0, std::ios_base::end);
  std::streampos size = file.tellg();
  if (size <= 0)
  {
    return false;
  }
  file.seekg(0, std::ios_base::beg);
  data.resize((size_t) size);
  file.read((char*) data.data(), (std::streamsize) data.size());
  return file.good();
}

void _LoadImage(const char* fileName, const std::vector<uint8_t>& ref,
                ImgioLoadFlags flags = ImgioLoadFlags::None)
{
  ImgioImage img;
  std::vector<uint8_t> fileData;

  REQUIRE(_ReadFile(fs::path(IMGIO_TESTENV_DIR) / fileName, fileData));
  CHECK_EQ(ImgioLoadImage(&fileData[0], fileData.size(), &img, flags), ImgioError::None);
  CHECK_EQ(img.data, ref);
}

void _LoadImageHdr(const char* fileName, const std::vector<half>& ref,
                   ImgioLoadFlags flags = ImgioLoadFlags::None)
{
  ImgioImage img;
  std::vector<uint8_t> fileData;

  REQUIRE(_ReadFile(fs::path(IMGIO_TESTENV_DIR) / fileName, fileData));
  REQUIRE_EQ(ImgioLoadImage(&fileData[0], fileData.size(), &img, flags | ImgioLoadFlags::KeepHdr), ImgioError::None);
  CHECK_EQ(img.format, ImgioFormat::RGBA16_FLOAT);

  std::vector<uint8_t> refBytes(ref.size() * sizeof(half));
  memcpy(refBytes.data(), ref.data(), refBytes.size());
  CHECK_EQ(img.data, refBytes);
}

static const std::vector<uint8_t> REF_4C = {255,   0,   0, 255,  // red
                                              0,   0, 255, 255,  // blue
                                            255, 255, 255, 255,  // white
                                              0, 255,   0, 255}; // green

static const std::vector<uint8_t> REF_4C_JPG = {254,   0,   0, 255,  // red
                                                  0,   0, 254, 255,  // blue
                                                255, 255, 255, 255,  // white
                                                  1, 255,   1, 255}; // green

static const std::vector<uint8_t> REF_3C_LONG = {127, 63, 191, 255, 127, 63, 191, 255,
                                                 127, 63, 191, 255, 127, 63, 191, 255};

TEST_CASE("Load.Png")
{
  _LoadImage("4c.png", REF_4C);
}

TEST_CASE("Load.Tiff")
{
  _LoadImage("4c.tiff", REF_4C);
}

TEST_CASE("Load.Exr")
{
  _LoadImage("4c.exr", REF_4C);
}

TEST_CASE("Load.Hdr")
{
  _LoadImage("4c.hdr", REF_4C);
}

TEST_CASE("Load.Jpg")
{
  _LoadImage("4c.jpg", REF_4C_JPG);
}

TEST_CASE("Load.Tga")
{
  _LoadImage("4c.tga", REF_4C);
}

// Channels are assigned to RGBA slots in canonical channel order (layer, well-known suffix
// priority, name), matching OIIO and thus Houdini (Karma). Missing channels stay zero and alpha
// defaults to one. With the 'greyscale to RGB' option, OIIO's gray_to_rgb behavior is applied: a
// lone channel is replicated to RGB, and a two-channel greyscale+alpha image becomes RRRA.
TEST_CASE("Load.ExrChannelMapping")
{
  struct ExrChannelTest
  {
    const char* fileName;
    std::vector<uint8_t> ref;
    std::vector<uint8_t> refGreyscale;
  };

  const ExrChannelTest cases[] = {
    // R
    { "1c_r_range.exr",
      {  0,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0, 255, 63,   0,   0, 255 },
      {  0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 63,  63,  63, 255 } },
    // Y (no special meaning; first channel)
    { "1c_y.exr",
      {127,   0,   0, 255,  63,   0,   0, 255,   0,   0,   0, 255, 255,   0,   0, 255 },
      {127, 127, 127, 255,  63,  63,  63, 255,   0,   0,   0, 255, 255, 255, 255, 255 } },
    // A (no special meaning; first channel)
    { "1c_a.exr",
      {127,   0,   0, 255,  63,   0,   0, 255,   0,   0,   0, 255, 255,   0,   0, 255 },
      {127, 127, 127, 255,  63,  63,  63, 255,   0,   0,   0, 255, 255, 255, 255, 255 } },
    // R G (B stays zero)
    { "2c_rg.exr",
      {191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255 },
      {191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255 } },
    // Z NX (canonical order: Z, NX)
    { "2c_z_nx.exr",
      {191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255 },
      {191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255 } },
    // RY BY (canonical order: BY, RY)
    { "2c_ry_by.exr",
      { 63, 191,   0, 255,  63, 191,   0, 255,  63, 191,   0, 255,  63, 191,   0, 255 },
      { 63, 191,   0, 255,  63, 191,   0, 255,  63, 191,   0, 255,  63, 191,   0, 255 } },
    // Y A (greyscale + alpha: promoted to RRRA by the option, like OIIO)
    { "2c_y_a.exr",
      {191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255, 191,  63,   0, 255 },
      {191, 191, 191,  63, 191, 191, 191,  63, 191, 191, 191,  63, 191, 191, 191,  63 } },
    // diffuse.R (layer prefix)
    { "1c_layer.exr",
      {127,   0,   0, 255,  63,   0,   0, 255,   0,   0,   0, 255, 255,   0,   0, 255 },
      {127, 127, 127, 255,  63,  63,  63, 255,   0,   0,   0, 255, 255, 255, 255, 255 } },
    // X Y Z (normal-map layer: X/Y/Z order instead of the default Y/Z/X)
    { "3c_xyz.exr",
      {191, 127,  63, 255, 191, 127,  63, 255, 191, 127,  63, 255, 191, 127,  63, 255 },
      {191, 127,  63, 255, 191, 127,  63, 255, 191, 127,  63, 255, 191, 127,  63, 255 } },
  };

  for (const ExrChannelTest& testCase : cases)
  {
    CAPTURE(testCase.fileName);
    _LoadImage(testCase.fileName, testCase.ref);
    _LoadImage(testCase.fileName, testCase.refGreyscale, ImgioLoadFlags::GreyscaleChannelToRgb);
  }
}

TEST_CASE("LoadHdr.Exr")
{
  _LoadImageHdr("4c.exr", { half(1.0f), half(0.0f), half(0.0f), half(1.0f),    // red
                            half(0.0f), half(0.0f), half(1.0f), half(1.0f),    // blue
                            half(1.0f), half(1.0f), half(1.0f), half(1.0f),    // white
                            half(0.0f), half(1.0f), half(0.0f), half(1.0f) }); // green
}

TEST_CASE("Load.ExrLongChannelNames")
{
  _LoadImage("3c_long.exr", REF_3C_LONG);
}

TEST_CASE("LoadHdr.ExrChannelMapping")
{
  // 1c_a: 0.5, 0.25, 0, 1 in output order.
  _LoadImageHdr("1c_a.exr", { half(0.5f),  half(0.0f),  half(0.0f),  half(1.0f),
                              half(0.25f), half(0.0f),  half(0.0f),  half(1.0f),
                              half(0.0f),  half(0.0f),  half(0.0f),  half(1.0f),
                              half(1.0f),  half(0.0f),  half(0.0f),  half(1.0f) });

  _LoadImageHdr("1c_a.exr", { half(0.5f),  half(0.5f),  half(0.5f),  half(1.0f),
                              half(0.25f), half(0.25f), half(0.25f), half(1.0f),
                              half(0.0f),  half(0.0f),  half(0.0f),  half(1.0f),
                              half(1.0f),  half(1.0f),  half(1.0f),  half(1.0f) },
                              ImgioLoadFlags::GreyscaleChannelToRgb);

  // 2c_z_nx: Z=0.75, NX=0.25; B stays zero.
  _LoadImageHdr("2c_z_nx.exr", { half(0.75f), half(0.25f), half(0.0f), half(1.0f),
                                 half(0.75f), half(0.25f), half(0.0f), half(1.0f),
                                 half(0.75f), half(0.25f), half(0.0f), half(1.0f),
                                 half(0.75f), half(0.25f), half(0.0f), half(1.0f) });
}
