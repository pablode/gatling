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

#include "MdlEntityResolver.h"

namespace sg
{
  MdlEntityResolver::MdlEntityResolver(mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> resolver)
    : m_resolver(resolver)
  {
  }

  mi::neuraylib::IMdl_resolved_module*
  MdlEntityResolver::resolve_module(const char* module_name,
                                    const char* owner_file_path,
                                    const char* owner_name,
                                    mi::Sint32 pos_line,
                                    mi::Sint32 pos_column,
                                    mi::neuraylib::IMdl_execution_context* context)
  {
    return m_resolver->resolve_module(module_name,
                                      owner_file_path,
                                      owner_name,
                                      pos_line,
                                      pos_column,
                                      context);
  }

  mi::neuraylib::IMdl_resolved_resource*
  MdlEntityResolver::resolve_resource(const char* file_path,
                                      const char* owner_file_path,
                                      const char* owner_name,
                                      mi::Sint32 pos_line,
                                      mi::Sint32 pos_column,
                                      mi::neuraylib::IMdl_execution_context* context)
  {
    return m_resolver->resolve_resource(file_path,
                                        owner_file_path,
                                        owner_name,
                                        pos_line,
                                        pos_column,
                                        context);
  }
}
