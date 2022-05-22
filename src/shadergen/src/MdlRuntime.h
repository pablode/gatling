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

#pragma once

#include "MdlLogger.h"
#include "MdlNeurayLoader.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_entity_resolver.h>
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
    mi::base::Handle<mi::neuraylib::IMdl_configuration> getConfig();
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> getImpExpApi();
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> getBackendApi();

  private:
    std::unique_ptr<MdlNeurayLoader> m_loader;

    mi::base::Handle<MdlLogger> m_logger;
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
    mi::base::Handle<mi::neuraylib::IDatabase> m_database;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::neuraylib::IMdl_configuration> m_config;
    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> m_entity_resolver;
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_factory;
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> m_backendApi;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_impExpApi;
  };
}
