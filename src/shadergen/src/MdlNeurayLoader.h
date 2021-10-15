#pragma once

#include <mi/base/handle.h>
#include <mi/neuraylib/ineuray.h>

namespace sg
{
  class MdlNeurayLoader
  {
  public:
    MdlNeurayLoader();
    ~MdlNeurayLoader();

  public:
    bool init(const char* resourcePath);

    mi::base::Handle<mi::neuraylib::INeuray> getNeuray() const;

  private:
    bool loadDso(const char* resourcePath);
    bool loadNeuray();
    void unloadDso();

  private:
    void* m_dsoHandle;
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
  };
}
