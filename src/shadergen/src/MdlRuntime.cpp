/******************************************************************************
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include "MdlRuntime.h"

#include <mi/mdl_sdk.h>

#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#endif

#include <string>

namespace sg
{
  MdlRuntime::MdlRuntime()
    : m_dsoHandle(nullptr)
    , m_neuray(nullptr)
  {
  }

  MdlRuntime::~MdlRuntime()
  {
    if (m_neuray)
    {
      m_neuray->shutdown();
      m_neuray->release();
    }
    if (m_dsoHandle)
    {
      unloadDso();
    }
  }

  bool MdlRuntime::init(const char* resourcePath)
  {
    if (!loadDso(resourcePath))
    {
      return false;
    }
    if (!loadNeuray())
    {
      return false;
    }
    return m_neuray->start() == 0;
  }

  mi::neuraylib::INeuray& MdlRuntime::getNeuray() const
  {
    return *m_neuray;
  }

  bool MdlRuntime::loadDso(const char* resourcePath)
  {
    std::string dsoFilename = std::string(resourcePath) + std::string("/libmdl_sdk" MI_BASE_DLL_FILE_EXT);

#ifdef MI_PLATFORM_WINDOWS
    HMODULE handle = LoadLibraryA(dsoFilename.c_str());
    if (!handle)
    {
      LPTSTR buffer = NULL;
      LPCTSTR message = TEXT("unknown failure");
      DWORD error_code = GetLastError();
      if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
      {
        message = buffer;
      }
      fprintf(stderr, "Failed to load library (%u): %s", error_code, message);
      if (buffer)
      {
        LocalFree(buffer);
      }
      return false;
    }
#else
    void* handle = dlopen(dsoFilename.c_str(), RTLD_LAZY);
    if (!handle)
    {
      fprintf(stderr, "%s\n", dlerror());
      return false;
    }
#endif

    m_dsoHandle = handle;
    return true;
  }

  bool MdlRuntime::loadNeuray()
  {
#ifdef MI_PLATFORM_WINDOWS
    void* symbol = GetProcAddress(reinterpret_cast<HMODULE>(m_dsoHandle), "mi_factory");
    if (!symbol)
    {
      LPTSTR buffer = NULL;
      LPCTSTR message = TEXT("unknown failure");
      DWORD error_code = GetLastError();
      if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
      {
        message = buffer;
      }
      fprintf(stderr, "GetProcAddress error (%u): %s", error_code, message);
      if (buffer)
      {
        LocalFree(buffer);
      }
      return false;
    }
#else
    void* symbol = dlsym(m_dsoHandle, "mi_factory");
    if (!symbol)
    {
      fprintf(stderr, "%s\n", dlerror());
      return false;
    }
#endif

    m_neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);

    if (m_neuray)
    {
      return true;
    }

    mi::base::Handle<mi::neuraylib::IVersion> version(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
    if (!version)
    {
      fprintf(stderr, "Error: Incompatible library.\n");
    }
    else
    {
      fprintf(stderr, "Error: Library version %s does not match header version %s.\n", version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
    }

    return false;
  }

  void MdlRuntime::unloadDso()
  {
#ifdef MI_PLATFORM_WINDOWS
    if (FreeLibrary(reinterpret_cast<HMODULE>(m_dsoHandle)))
    {
      return;
    }
    LPTSTR buffer = 0;
    LPCTSTR message = TEXT("unknown failure");
    DWORD error_code = GetLastError();
    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
        MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
    {
        message = buffer;
    }
    fprintf(stderr, "Failed to unload library (%u): %s", error_code, message);
    if (buffer)
    {
      LocalFree(buffer);
    }
#else
    if (dlclose(m_dsoHandle) != 0)
    {
      printf( "%s\n", dlerror());
    }
#endif
  }
}
