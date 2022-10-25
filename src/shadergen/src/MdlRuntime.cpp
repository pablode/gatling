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
#include "MdlEntityResolver.h"

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
    if (m_neuray)
    {
      m_neuray->shutdown();
    }
  }

  bool MdlRuntime::init(const char* resourcePath)
  {
    m_loader = std::make_unique<MdlNeurayLoader>();
    if (!m_loader->init(resourcePath))
    {
      return false;
    }

    m_neuray = mi::base::Handle<mi::neuraylib::INeuray>(m_loader->getNeuray());

    m_config = mi::base::Handle<mi::neuraylib::IMdl_configuration>(m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    m_logger = mi::base::Handle<MdlLogger>(new MdlLogger());
    m_config->set_logger(m_logger.get());

    if (m_neuray->start() != 0)
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "Unable to start Neuray");
      return false;
    }

    auto oldEntityResolver = mi::base::Handle<mi::neuraylib::IMdl_entity_resolver>(m_config->get_entity_resolver());
    m_entity_resolver = mi::base::Handle<MdlEntityResolver>(new MdlEntityResolver(oldEntityResolver));
    m_config->set_entity_resolver(m_entity_resolver.get());

    m_database = mi::base::Handle<mi::neuraylib::IDatabase>(m_neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope(m_database->get_global_scope());
    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(scope->create_transaction());

    m_factory = mi::base::Handle<mi::neuraylib::IMdl_factory>(m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    m_impExpApi = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    m_backendApi = mi::base::Handle<mi::neuraylib::IMdl_backend_api>(m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
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

  mi::base::Handle<mi::neuraylib::IMdl_configuration> MdlRuntime::getConfig()
  {
    return m_config;
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
