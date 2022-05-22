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

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/imdl_entity_resolver.h>

namespace sg
{
  class MdlEntityResolver : public mi::base::Interface_implement<mi::neuraylib::IMdl_entity_resolver>
  {
  public:
    MdlEntityResolver(mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> resolver);

  public:
    mi::neuraylib::IMdl_resolved_module*
    resolve_module(const char* module_name,
                   const char* owner_file_path,
                   const char* owner_name,
                   mi::Sint32 pos_line,
                   mi::Sint32 pos_column,
                   mi::neuraylib::IMdl_execution_context* context = 0) override;

    mi::neuraylib::IMdl_resolved_resource*
    resolve_resource(const char* file_path,
                     const char* owner_file_path,
                     const char* owner_name,
                     mi::Sint32 pos_line,
                     mi::Sint32 pos_column,
                     mi::neuraylib::IMdl_execution_context* context = 0) override;

  private:
    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> m_resolver;
  };
}
