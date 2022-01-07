#pragma once

#include "MdlLogger.h"
#include "MdlNeurayLoader.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imdl_factory.h>

#include <memory>

namespace sg
{
  class MdlRuntime
  {
  public:
    MdlRuntime();
    ~MdlRuntime();

  public:
    bool init(const char* resourcePath,
              const char* mtlxmdlPath);

    mi::base::Handle<MdlLogger> getLogger();
    mi::base::Handle<mi::neuraylib::IDatabase> getDatabase();
    mi::base::Handle<mi::neuraylib::ITransaction> getTransaction();
    mi::base::Handle<mi::neuraylib::IMdl_factory> getFactory();
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> getImpExpApi();
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> getBackendApi();

  private:
    std::unique_ptr<MdlNeurayLoader> m_loader;

    mi::base::Handle<MdlLogger> m_logger;
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
    mi::base::Handle<mi::neuraylib::IDatabase> m_database;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_factory;
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> m_backendApi;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_impExpApi;
  };
}
