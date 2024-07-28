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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-function"
#include <mi/base/handle.h>
#include <mi/neuraylib/ineuray.h>
#pragma clang diagnostic pop

#include <string_view>

namespace gtl
{
  class McMdlNeurayLoader
  {
  public:
    McMdlNeurayLoader();
    ~McMdlNeurayLoader();

  public:
    bool init(std::string_view resourcePath);

    mi::base::Handle<mi::neuraylib::INeuray> getNeuray() const;

  private:
    void* m_dsoHandle;
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
  };
}
