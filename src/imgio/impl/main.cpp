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
  data.resize(file.tellg());
  file.seekg(0, std::ios_base::beg);
  file.read((char*) &data[0], data.size());
  return file.good() && data.size() > 0;
}

void _LoadOriented(const char* fileName, const std::vector<uint8_t>& ref)
{
  ImgioImage img;
  std::vector<uint8_t> fileData;

  REQUIRE(_ReadFile(fs::path(IMGIO_TESTENV_DIR) / fileName, fileData));
  CHECK_EQ(ImgioLoadImage(&fileData[0], fileData.size(), &img), ImgioError::None);
  CHECK_EQ(img.data, ref);
}

static const std::vector<uint8_t> REF_4C = {255,   0,   0, 255,  // red
                                              0,   0, 255, 255,  // blue
                                            255, 255, 255, 255,  // white
                                              0, 255,   0, 255}; // green

static const std::vector<uint8_t> REF_4C_JPG = {254,   0,   0, 255,  // red
                                                  0,   0, 254, 255,  // blue
                                                255, 255, 255, 255,  // white
                                                  1, 255,   1, 255}; // green

TEST_CASE("LoadOriented.Png")
{
  _LoadOriented("4c.png", REF_4C);
}

TEST_CASE("LoadOriented.Tiff")
{
  _LoadOriented("4c.tiff", REF_4C);
}

TEST_CASE("LoadOriented.Exr")
{
  _LoadOriented("4c.exr", REF_4C);
}

TEST_CASE("LoadOriented.Hdr")
{
  _LoadOriented("4c.hdr", REF_4C);
}

TEST_CASE("LoadOriented.Jpg")
{
  _LoadOriented("4c.jpg", REF_4C_JPG);
}

TEST_CASE("LoadOriented.Tga")
{
  _LoadOriented("4c.tga", REF_4C);
}
