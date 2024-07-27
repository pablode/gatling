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

#include "MdlNeurayLoader.h"

#include <mi/mdl_sdk.h>

#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#endif

#include <string>
#include <gtl/gb/Log.h>

#define GTL_RECOMMENDED_NEURAYLIB_VERSION 51
#define GTL_RECOMMENDED_NEURAYLIB_VERSION_STRING "2023.0.4"
#define GTL_LATEST_TESTED_NEURAYLIB_VERSION 52

static_assert(MI_NEURAYLIB_API_VERSION >= 48, "MDL SDK version is too old!");
static_assert(MI_NEURAYLIB_API_VERSION < 52, "2023.1.X MDL SDK has crash issues - use 2023.0.4 instead!");
static_assert(MI_NEURAYLIB_API_VERSION <= GTL_LATEST_TESTED_NEURAYLIB_VERSION, "Untested MDL SDK version!");

namespace
{
  void* _LoadDso(std::string_view libDir)
  {
    std::string dsoFilename = std::string(libDir) + std::string("/libmdl_sdk" MI_BASE_DLL_FILE_EXT);

#ifdef MI_PLATFORM_WINDOWS
    HMODULE handle = LoadLibraryA(dsoFilename.c_str());
    if (!handle)
    {
      LPTSTR buffer = NULL;
      LPCTSTR message = TEXT("unknown error");
      DWORD errorCode = GetLastError();
      if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS, 0, errorCode,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
      {
        message = buffer;
      }
      GB_ERROR("failed to load MDL library ({}): {}", errorCode, message);
      if (buffer)
      {
        LocalFree(buffer);
      }
      return nullptr;
    }
#else
    void* handle = dlopen(dsoFilename.c_str(), RTLD_LAZY);
    if (!handle)
    {
      const char* error = dlerror();
      if (!error)
      {
        error = "unknown error";
      }
      GB_ERROR("failed to load MDL library: {}", error);
      return nullptr;
    }
#endif

    return handle;
  }

  void _UnloadDso(void* handle)
  {
#ifdef MI_PLATFORM_WINDOWS
    if (FreeLibrary(reinterpret_cast<HMODULE>(handle)))
    {
      return;
    }
    LPTSTR buffer = 0;
    LPCTSTR message = TEXT("unknown error");
    DWORD errorCode = GetLastError();
    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS, 0, errorCode,
        MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
    {
        message = buffer;
    }
    GB_ERROR("failed to unload MDL library ({}): {}", errorCode, message);
    if (buffer)
    {
      LocalFree(buffer);
    }
#else
    if (dlclose(handle) == 0)
    {
      return;
    }
    const char* error = dlerror();
    if (!error)
    {
      error = "unknown error";
    }
    GB_ERROR("failed to unload MDL library: {}", error);
#endif
  }

  mi::base::Handle<mi::neuraylib::INeuray> _LoadNeuray(void* dsoHandle)
  {
#ifdef MI_PLATFORM_WINDOWS
    void* symbol = GetProcAddress(reinterpret_cast<HMODULE>(dsoHandle), "mi_factory");
    if (!symbol)
    {
      LPTSTR buffer = NULL;
      LPCTSTR message = TEXT("unknown error");
      DWORD errorCode = GetLastError();
      if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS, 0, errorCode,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
      {
        message = buffer;
      }
      GB_ERROR("failed to locate MDL library entry point ({}): {}", errorCode, message);
      if (buffer)
      {
        LocalFree(buffer);
      }
      return {};
    }
#else
    void* symbol = dlsym(dsoHandle, "mi_factory");
    if (!symbol)
    {
      const char* error = dlerror();
      if (!error)
      {
        error = "unknown error";
      }
      GB_ERROR("failed to locate MDL library entry point: {}", error);
      return {};
    }
#endif

    mi::base::Handle<mi::neuraylib::IVersion> version(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
    if (!version)
    {
      GB_ERROR("failed to load MDL library: invalid library");
      return {};
    }
    else if (strcmp(version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING) != 0)
    {
      GB_ERROR("failed to load MDL library: version {} does not match header version {}",
        version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
      return {};
    }

    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol));
    if (!neuray.is_valid_interface())
    {
      GB_ERROR("failed to load MDL library: unknown error");
      return {};
    }

    if (MI_NEURAYLIB_API_VERSION != GTL_RECOMMENDED_NEURAYLIB_VERSION)
    {
      GB_WARN("not using recommended MDL SDK version {}", GTL_RECOMMENDED_NEURAYLIB_VERSION_STRING);
    }

    return neuray;
  }
}

namespace gtl
{
  McMdlNeurayLoader::McMdlNeurayLoader()
    : m_dsoHandle(nullptr)
    , m_neuray(nullptr)
  {
  }

  McMdlNeurayLoader::~McMdlNeurayLoader()
  {
    m_neuray.reset();
    if (m_dsoHandle)
    {
      _UnloadDso(m_dsoHandle);
    }
  }

  bool McMdlNeurayLoader::init(std::string_view libDir)
  {
    m_dsoHandle = _LoadDso(libDir);
    if (!m_dsoHandle)
    {
      return false;
    }

    m_neuray = _LoadNeuray(m_dsoHandle);
    if (!m_neuray)
    {
      return false;
    }

    return true;
  }

  mi::base::Handle<mi::neuraylib::INeuray> McMdlNeurayLoader::getNeuray() const
  {
    return m_neuray;
  }
}
