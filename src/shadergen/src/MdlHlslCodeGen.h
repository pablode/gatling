#pragma once

#include <mi/base/handle.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_execution_context.h>

#include <stdint.h>
#include <string>
#include <vector>

#include "MdlRuntime.h"
#include "MdlLogger.h"

namespace sg
{
  class MdlHlslCodeGen
  {
  public:
    bool init(MdlRuntime& runtime);

    bool translate(const std::vector<const mi::neuraylib::ICompiled_material*>& materials,
                   std::string& hlslSrc);

  private:
    bool appendMaterialToLinkUnit(uint32_t idx,
                                  const mi::neuraylib::ICompiled_material* compiledMaterial,
                                  mi::neuraylib::ILink_unit* linkUnit);

  private:
    mi::base::Handle<MdlLogger> m_logger;
    mi::base::Handle<mi::neuraylib::IMdl_backend> m_backend;
    mi::base::Handle<mi::neuraylib::IDatabase> m_database;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_context;
  };
}
