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

#include "MdlEntityResolver.h"

#include <string>
#include <string.h>
#include <assert.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace gtl
{
  // adapted from MDL SDK compilercore_file_resolution.cpp
  std::string _ModuleNameToUrl(const char* inputName)
  {
    std::string inputUrl;
    while (true)
    {
      char const* p = strstr(inputName, "::");

      if (p == NULL) {
        inputUrl.append(inputName);
        return inputUrl;
      }
      inputUrl.append(inputName, p - inputName);
      inputUrl.append("/");
      inputName = p + 2;
    }
  }

  class McMdlResolvedModule
    : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_module>
  {
  public:
    McMdlResolvedModule(mi::base::Handle<mi::neuraylib::IMdl_impexp_api> impExpApi,
                        std::string_view moduleName,
                        std::string_view filePath)
      : m_impExpApi(impExpApi)
      , m_moduleName(moduleName)
      , m_filePath(filePath)
    {
    }

    const char* get_module_name() const { return m_moduleName.c_str(); }

    const char* get_filename() const { return m_filePath.c_str(); }

    mi::neuraylib::IReader* create_reader() const
    {
      return m_impExpApi->create_reader(get_filename());
    }

  private:
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_impExpApi;
    std::string m_moduleName;
    std::string m_filePath;
  };

  McMdlEntityResolver::McMdlEntityResolver(mi::base::Handle<mi::neuraylib::IMdl_impexp_api> impExpApi,
                                           mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> standardResolver)
    : m_impExpApi(impExpApi)
    , m_standardResolver(standardResolver)
  {
  }

  mi::neuraylib::IMdl_resolved_module* McMdlEntityResolver::resolve_module(const char* module_name,
                                                                           const char* owner_file_path,
                                                                           const char* owner_name,
                                                                           mi::Sint32 pos_line,
                                                                           mi::Sint32 pos_column,
                                                                           mi::neuraylib::IMdl_execution_context* context)
  {
    constexpr static const char* MTLX_SUBSTRING = "materialx";
    constexpr static const char* MDL_FILE_EXT = ".mdl";

    const mi::base::IInterface* boxedUserData = nullptr;
    mi::Sint32 result = -1;
    if (context)
    {
      result = context->get_option("user_data", &boxedUserData);
    }

    mi::base::Handle<const McMdlEntityResolverUserData> userData((boxedUserData && result == 0) ?
      boxedUserData->get_interface<McMdlEntityResolverUserData>() : nullptr);

    if (userData)
    {
      // Weird edge case found in some of the assets
      const char* newModuleName = module_name;
      if (strlen(newModuleName) > 2 && newModuleName[0] == '.' && newModuleName[1] == ':' && newModuleName[2] == ':')
      {
        newModuleName++;
      }

      std::string moduleUrl = _ModuleNameToUrl(newModuleName);
      std::string filePath = std::string(userData->dirPrefix) + moduleUrl + MDL_FILE_EXT;

      // Only resolve MDL file if it actually exists - fall back to standard
      // resolver with registered system/user/gatling search paths otherwise.
      if (fs::exists(fs::path(filePath)))
      {
        return new McMdlResolvedModule(m_impExpApi, newModuleName, filePath);
      }
    }

    // Fall back to default resolver
    return m_standardResolver->resolve_module(module_name, owner_file_path, owner_name, pos_line, pos_column, context);
  }

  mi::neuraylib::IMdl_resolved_resource* McMdlEntityResolver::resolve_resource(const char* file_path,
                                                                               const char* owner_file_path,
                                                                               const char* owner_name,
                                                                               mi::Sint32 pos_line,
                                                                               mi::Sint32 pos_column,
                                                                               mi::neuraylib::IMdl_execution_context* context)
  {
    return m_standardResolver->resolve_resource(file_path, owner_file_path, owner_name, pos_line, pos_column, context);
  }
}
