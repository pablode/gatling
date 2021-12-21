#pragma once

#include <string>

#include <mi/base/handle.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>

#include "MdlRuntime.h"

namespace sg
{
  class MdlMaterialCompiler
  {
  public:
    MdlMaterialCompiler(MdlRuntime& runtime);

  public:
    bool compileMaterial(const std::string& src,
                         const std::string& identifier,
                         mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial);

  private:
    bool createModule(mi::neuraylib::IMdl_execution_context* context,
                      const char* moduleName,
                      const char* mdlSrc);

    bool createCompiledMaterial(mi::neuraylib::IMdl_execution_context* context,
                                const char* moduleName,
                                const std::string& identifier,
                                mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial);

  private:
    mi::base::Handle<MdlLogger> m_logger;
    mi::base::Handle<mi::neuraylib::IDatabase> m_database;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::neuraylib::IMdl_configuration> m_config;
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_factory;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_impExpApi;
  };
}
