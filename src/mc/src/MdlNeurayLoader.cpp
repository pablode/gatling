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

#define MI_NEURAYLIB_LATEST_VERSION 51

static_assert(MI_NEURAYLIB_API_VERSION >= 48, "MDL SDK version is too old!");
static_assert(MI_NEURAYLIB_API_VERSION <= MI_NEURAYLIB_LATEST_VERSION, "Untested MDL SDK version!");

namespace
{
  void* _LoadDso(std::string_view resourcePath)
  {
    std::string dsoFilename = std::string(resourcePath) + std::string("/libmdl_sdk" MI_BASE_DLL_FILE_EXT);

#ifdef MI_PLATFORM_WINDOWS
    HMODULE handle = LoadLibraryA(dsoFilename.c_str());
    if (!handle)
    {
      LPTSTR buffer = NULL;
      LPCTSTR message = TEXT("unknown error");
      DWORD error_code = GetLastError();
      if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
      {
        message = buffer;
      }
      fprintf(stderr, "Failed to load MDL library (%u): %s\n", error_code, message);
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
      fprintf(stderr, "Failed to load MDL library: %s\n", error);
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
    DWORD error_code = GetLastError();
    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
        MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
    {
        message = buffer;
    }
    fprintf(stderr, "Failed to unload MDL library (%u): %s\n", error_code, message);
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
    fprintf(stderr, "Failed to unload MDL library: %s\n", error);
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
      fprintf(stderr, "Failed to locate MDL library entry point (%u): %s\n", errorCode, message);
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
      fprintf(stderr, "Failed to locate MDL library entry point: %s\n", error);
      return {};
    }
#endif

    mi::base::Handle<mi::neuraylib::IVersion> version(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
    if (!version)
    {
      fprintf(stderr, "Failed to load MDL library: invalid library\n");
      return {};
    }
    else if (strcmp(version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING) != 0)
    {
      fprintf(stderr, "Failed to load MDL library: version %s does not match header version %s\n",
        version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
      return {};
    }

    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol));
    if (!neuray.is_valid_interface())
    {
      fprintf(stderr, "Failed to load MDL library: unknown error\n");
      return {};
    }

    if (MI_NEURAYLIB_API_VERSION != MI_NEURAYLIB_LATEST_VERSION)
    {
      fprintf(stderr, "Warning: not using the latest MDL SDK - update for bugfixes\n");
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

  bool McMdlNeurayLoader::init(std::string_view resourcePath)
  {
    m_dsoHandle = _LoadDso(resourcePath);
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
