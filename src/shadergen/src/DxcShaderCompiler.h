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

#pragma once

#include "IShaderCompiler.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <atlbase.h>
#include <dxc/dxcapi.h>

namespace sg
{
  class DxcShaderCompiler : public IShaderCompiler
  {
  public:
    DxcShaderCompiler(std::string_view shaderPath);

  public:
    bool init() override;

    bool compileHlslToSpv(std::string_view source,
                          std::string_view filePath,
                          std::string_view entryPoint,
                          std::vector<uint8_t>& spv) override;

  private:
    CComPtr<IDxcCompiler3> m_compiler;
    CComPtr<IDxcIncludeHandler> m_includeHandler;
    CComPtr<IDxcUtils> m_utils;
  };
}
