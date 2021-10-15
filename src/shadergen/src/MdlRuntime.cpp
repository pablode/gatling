#include "MdlRuntime.h"

#include <mi/mdl_sdk.h>

namespace sg
{
  MdlRuntime::MdlRuntime()
  {
  }

  MdlRuntime::~MdlRuntime()
  {
    if (m_transaction)
    {
      m_transaction->commit();
    }
  }

  bool MdlRuntime::init(const char* resourcePath,
                        const char* mtlxmdlPath)
  {
    m_loader = std::make_unique<MdlNeurayLoader>();
    if (!m_loader->init(resourcePath))
    {
      return false;
    }

    mi::base::Handle<mi::neuraylib::INeuray> neuray(m_loader->getNeuray());
    mi::base::Handle<mi::neuraylib::IMdl_configuration> config(neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

    m_logger = mi::base::Handle<MdlLogger>(new MdlLogger());
    config->set_logger(m_logger.get());

    if (config->add_mdl_path(mtlxmdlPath))
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "MaterialX MDL file path not found, translation not possible");
      return false;
    }

    m_database = mi::base::Handle<mi::neuraylib::IDatabase>(neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope(m_database->get_global_scope());
    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(scope->create_transaction());

    m_factory = mi::base::Handle<mi::neuraylib::IMdl_factory>(neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    m_impExpApi = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    m_backendApi = mi::base::Handle<mi::neuraylib::IMdl_backend_api>(neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
    return true;
  }

  mi::base::Handle<MdlLogger> MdlRuntime::getLogger()
  {
    return m_logger;
  }

  mi::base::Handle<mi::neuraylib::IDatabase> MdlRuntime::getDatabase()
  {
    return m_database;
  }

  mi::base::Handle<mi::neuraylib::ITransaction> MdlRuntime::getTransaction()
  {
    return m_transaction;
  }

  mi::base::Handle<mi::neuraylib::IMdl_factory> MdlRuntime::getFactory()
  {
    return m_factory;
  }

  mi::base::Handle<mi::neuraylib::IMdl_impexp_api> MdlRuntime::getImpExpApi()
  {
    return m_impExpApi;
  }

  mi::base::Handle<mi::neuraylib::IMdl_backend_api> MdlRuntime::getBackendApi()
  {
    return m_backendApi;
  }
}
