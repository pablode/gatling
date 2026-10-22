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

#include <mi/base/config.h>

#include <cassert>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

namespace fs = std::filesystem;

namespace gtl
{
  namespace
  {
    constexpr const char* MDL_FILE_EXT = ".mdl";

    std::string _GetDirectorySeparator()
    {
#ifdef MI_PLATFORM_WINDOWS
      return "\\";
#else
      return "/";
#endif
    }

    std::string _ReplaceAll(std::string str, std::string_view from, std::string_view to)
    {
      size_t pos = 0;
      while ((pos = str.find(from.data(), pos, from.size())) != std::string::npos)
      {
        str.replace(pos, from.size(), to.data(), to.size());
        pos += to.size();
      }
      return str;
    }

    bool _IsAbsoluteModuleName(const char* moduleName)
    {
      return moduleName && moduleName[0] == ':' && moduleName[1] == ':';
    }

    struct AbsoluteModule
    {
      std::string moduleName;
      std::string filePath;
    };

    // Adapted from the MDL SDK entity resolver example. Computes the absolute
    // module name and filename for a relative module name and its owner.
    AbsoluteModule _GetAbsoluteModule(const char* moduleName,
                                      const char* ownerName,
                                      const char* ownerFilePath)
    {
      assert(moduleName && ownerName && ownerFilePath);
      assert(!_IsAbsoluteModuleName(moduleName));

      const std::string separator = _GetDirectorySeparator();
      std::string nameStart = ownerName;
      std::string fileNameStart = fs::path(ownerFilePath).make_preferred().string();
      std::string end = moduleName;

      // Remove the owner module name and owner filename from the starts.
      size_t namePos = nameStart.rfind("::");
      size_t fileNamePos = fileNameStart.rfind(separator);
      if (namePos == std::string::npos || fileNamePos == std::string::npos)
      {
        return {};
      }
      nameStart = nameStart.substr(0, namePos);
      fileNameStart = fileNameStart.substr(0, fileNamePos);

      // Strip ".::" prefix.
      if (end.substr(0, 3) == ".::")
      {
        end = end.substr(3);
      }

      // Strip the last component from both starts for each "..::" prefix.
      while (end.substr(0, 4) == "..::")
      {
        namePos = nameStart.rfind("::");
        fileNamePos = fileNameStart.rfind(separator);
        if (namePos == std::string::npos || fileNamePos == std::string::npos)
        {
          return {};
        }
        end = end.substr(4);
        nameStart = nameStart.substr(0, namePos);
        fileNameStart = fileNameStart.substr(0, fileNamePos);
      }

      AbsoluteModule result;
      result.moduleName = nameStart + "::" + end;

      std::string fileNameEnd = _ReplaceAll(end, "::", separator);
      result.filePath = fileNameStart + separator + fileNameEnd + MDL_FILE_EXT;

      return result;
    }

    // Maps an absolute module name to a file below the entry module's directory.
    std::string _GetAbsoluteModuleFilePath(const std::string& rootDir, const char* moduleName)
    {
      const std::string separator = _GetDirectorySeparator();
      std::string suffix = _ReplaceAll(moduleName, "::", separator);
      return rootDir + suffix + MDL_FILE_EXT;
    }

    void _AddErrorMessage(mi::neuraylib::IMdl_execution_context* context,
                          const char* identifier,
                          const std::string& reason)
    {
      if (!context)
      {
        return;
      }

      std::string message = "Failed to resolve \"";
      message += identifier ? identifier : "";
      message += "\" (";
      message += reason;
      message += ")";

      context->add_message(mi::neuraylib::IMessage::MSG_INTEGRATION,
                           mi::base::MESSAGE_SEVERITY_ERROR,
                           -1,
                           message.c_str());
    }

    bool _PathsAreEquivalent(const char* pathA, const char* pathB)
    {
      if (!pathA || !pathB)
      {
        return false;
      }

      std::error_code errorCode;
      bool equivalent = fs::equivalent(pathA, pathB, errorCode);
      if (!errorCode)
      {
        return equivalent;
      }

      errorCode.clear();
      fs::path canonicalA = fs::weakly_canonical(pathA, errorCode);
      if (errorCode)
      {
        return fs::path(pathA).lexically_normal() == fs::path(pathB).lexically_normal();
      }

      fs::path canonicalB = fs::weakly_canonical(pathB, errorCode);
      if (errorCode)
      {
        return fs::path(pathA).lexically_normal() == fs::path(pathB).lexically_normal();
      }

      return canonicalA == canonicalB;
    }

    std::string _GetRootPackage(const std::string& moduleName)
    {
      size_t pos = moduleName.rfind("::");
      return pos != std::string::npos ? moduleName.substr(0, pos) : std::string();
    }

    // Strips the per-root synthetic package so that the module can be looked
    // up through the configured search paths.
    std::string _GetCanonicalModuleName(const std::string& moduleName, const std::string& rootPackage)
    {
      if (rootPackage.empty())
      {
        return moduleName;
      }

      std::string prefix = rootPackage + "::";
      if (moduleName.compare(0, prefix.size(), prefix) == 0)
      {
        return "::" + moduleName.substr(prefix.size());
      }

      return moduleName;
    }

    // Returns the filename that the search paths resolve the absolute module
    // name to, or an empty string if there is no such module.
    std::string _FindCanonicalModule(mi::neuraylib::IMdl_entity_resolver* resolver,
                                     const std::string& absoluteModuleName)
    {
      mi::base::Handle<mi::neuraylib::IMdl_resolved_module> module(
        resolver->resolve_module(absoluteModuleName.c_str(), nullptr, nullptr, 0, 0, nullptr));

      if (!module)
      {
        return {};
      }

      const char* filename = module->get_filename();
      return filename ? std::string(filename) : std::string();
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
    const mi::base::IInterface* boxedUserData = nullptr;
    mi::Sint32 result = -1;
    if (context)
    {
      result = context->get_option("user_data", &boxedUserData);
    }

    mi::base::Handle<const McMdlEntityResolverUserData> userData((boxedUserData && result == 0) ?
      boxedUserData->get_interface<McMdlEntityResolverUserData>() : nullptr);

    if (userData && module_name)
    {
      // The top-level module of a file-backed compilation is mapped to the
      // exact file it was requested for.
      if (userData->entryModuleName == module_name)
      {
        return new McMdlResolvedModule(m_impExpApi, userData->entryModuleName, userData->entryFilePath);
      }

      std::string absoluteModuleName;
      std::string filePath;
      bool isRelativeImport = false;

      if (_IsAbsoluteModuleName(module_name))
      {
        absoluteModuleName = module_name;
        filePath = _GetAbsoluteModuleFilePath(userData->rootDir, module_name);
      }
      else if (owner_name && owner_file_path)
      {
        isRelativeImport = true;

        AbsoluteModule absoluteModule = _GetAbsoluteModule(module_name, owner_name, owner_file_path);
        absoluteModuleName = absoluteModule.moduleName;
        filePath = absoluteModule.filePath;
      }

      if (!absoluteModuleName.empty() && !filePath.empty())
      {
        // Only resolve the MDL file if it actually exists - fall back to the
        // standard resolver with the registered search paths otherwise.
        std::error_code errorCode;
        bool exists = fs::exists(fs::path(filePath), errorCode);

        if (!errorCode && exists)
        {
          // Check 1 from section 2.2 of the MDL specification: a relative
          // import must not resolve to a file that is shadowed by a module in
          // the search paths.
          if (isRelativeImport)
          {
            std::string rootPackage = _GetRootPackage(userData->entryModuleName);
            std::string canonicalModuleName = _GetCanonicalModuleName(absoluteModuleName, rootPackage);
            std::string canonicalFile = _FindCanonicalModule(m_standardResolver.get(), canonicalModuleName);

            if (!canonicalFile.empty() && !_PathsAreEquivalent(canonicalFile.c_str(), filePath.c_str()))
            {
              std::string reason = "shadowing rule: \"" + canonicalFile + "\" shadows \"" + filePath + "\"";
              _AddErrorMessage(context, module_name, reason);
              return nullptr;
            }
          }

          return new McMdlResolvedModule(m_impExpApi, absoluteModuleName, filePath);
        }
      }
    }

    // Fall back to the default resolver. This also handles built-in modules and
    // modules located in the configured system/user/application search paths.
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
