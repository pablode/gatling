#pragma once

#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imdl_backend.h>

#include <stdint.h>
#include <string>
#include <vector>

namespace sg
{
  struct SourceIdentifierPair
  {
    std::string src;
    std::string identifier;
  };

  class MdlHlslCodeGen
  {
  public:
    bool init(mi::neuraylib::INeuray& neuray,
              const char* mtlxmdlPath);

    bool translate(const std::vector<const SourceIdentifierPair*>& input,
                   std::string& hlslSrc);

  private:
    bool appendModuleToLinkUnit(const SourceIdentifierPair& sourceAndIdentifier,
                                uint32_t idx,
                                mi::neuraylib::ITransaction* transaction,
                                mi::neuraylib::ILink_unit* linkUnit);

    void printContextMessages();

  private:
    mi::base::Handle<mi::base::ILogger> m_logger;
    mi::base::Handle<mi::neuraylib::IDatabase> m_mdlDatabase;
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_mdlFactory;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_mdlContext;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_mdlImpExpApi;
    mi::base::Handle<mi::neuraylib::IMdl_backend> m_mdlBackend;
  };
}
