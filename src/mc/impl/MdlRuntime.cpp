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

#include "MdlRuntime.h"

#include "MdlLogger.h"
#include "MdlNeurayLoader.h"

#include <mi/mdl_sdk.h>

namespace gtl
{
  McMdlRuntime::McMdlRuntime()
  {
  }

  McMdlRuntime::~McMdlRuntime()
  {
    if (m_transaction)
    {
      m_transaction->commit();
    }
  }

  bool McMdlRuntime::init(std::string_view libDir)
  {
    m_loader = std::make_shared<McMdlNeurayLoader>();
    if (!m_loader->init(libDir))
    {
      return false;
    }

    m_neuray = mi::base::Handle<mi::neuraylib::INeuray>(m_loader->getNeuray());

    m_config = mi::base::Handle<mi::neuraylib::IMdl_configuration>(m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

    m_logger = mi::base::Handle<McMdlLogger>(new McMdlLogger());
#if MI_NEURAYLIB_API_VERSION < 52
    m_config->set_logger(m_logger.get());
#else
    mi::base::Handle<mi::neuraylib::ILogging_configuration> loggingConfig(m_neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
    loggingConfig->set_receiving_logger(m_logger.get());
#endif

    if (m_neuray->start() != 0)
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "Unable to start Neuray");
      return false;
    }

    m_database = mi::base::Handle<mi::neuraylib::IDatabase>(m_neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope(m_database->get_global_scope());
    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(scope->create_transaction());

    m_factory = mi::base::Handle<mi::neuraylib::IMdl_factory>(m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    m_impExpApi = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    m_backendApi = mi::base::Handle<mi::neuraylib::IMdl_backend_api>(m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
    return true;
  }

  mi::base::Handle<McMdlLogger> McMdlRuntime::getLogger()
  {
    return m_logger;
  }

  mi::base::Handle<mi::neuraylib::IDatabase> McMdlRuntime::getDatabase()
  {
    return m_database;
  }

  mi::base::Handle<mi::neuraylib::ITransaction> McMdlRuntime::getTransaction()
  {
    return m_transaction;
  }

  mi::base::Handle<mi::neuraylib::IMdl_factory> McMdlRuntime::getFactory()
  {
    return m_factory;
  }

  mi::base::Handle<mi::neuraylib::IMdl_configuration> McMdlRuntime::getConfig()
  {
    return m_config;
  }

  mi::base::Handle<mi::neuraylib::IMdl_impexp_api> McMdlRuntime::getImpExpApi()
  {
    return m_impExpApi;
  }

  mi::base::Handle<mi::neuraylib::IMdl_backend_api> McMdlRuntime::getBackendApi()
  {
    return m_backendApi;
  }
}
