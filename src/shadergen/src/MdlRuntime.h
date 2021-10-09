#pragma once

#include <mi/neuraylib/ineuray.h>

namespace sg
{
  class MdlRuntime
  {
  public:
    MdlRuntime();
    ~MdlRuntime();

  public:
    bool init(const char* resourcePath);

    mi::neuraylib::INeuray& getNeuray() const;

  private:
    bool loadDso(const char* resourcePath);
    bool loadNeuray();
    void unloadDso();

  private:
    void* m_dsoHandle;
    mi::neuraylib::INeuray* m_neuray;
  };
}
