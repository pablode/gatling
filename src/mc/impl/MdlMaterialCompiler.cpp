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

#include "MdlMaterialCompiler.h"

#include "MdlRuntime.h"
#include "MdlEntityResolver.h"

#include <gtl/gb/Fmt.h>

#include <mi/mdl_sdk.h>

#include <atomic>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

namespace
{
  std::atomic_uint32_t s_idCounter(0);

  std::string _MakeModuleName(std::string_view identifier)
  {
    uint32_t uniqueId = ++s_idCounter;
    return GB_FMT("::gatling::{}_{}", uniqueId, identifier);
  }
}

namespace gtl
{
  McMdlMaterialCompiler::McMdlMaterialCompiler(McMdlRuntime& runtime)
  {
    m_logger = mi::base::Handle<McMdlLogger>(runtime.getLogger());
    m_database = mi::base::Handle<mi::neuraylib::IDatabase>(runtime.getDatabase());
    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(runtime.getTransaction());
    m_config = mi::base::Handle<mi::neuraylib::IMdl_configuration>(runtime.getConfig());
    m_factory = mi::base::Handle<mi::neuraylib::IMdl_factory>(runtime.getFactory());
    m_impExpApi = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(runtime.getImpExpApi());
  }

  bool McMdlMaterialCompiler::compileFromString(std::string_view srcStr,
                                                std::string_view identifier,
                                                mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    std::string moduleName = _MakeModuleName(identifier);

    auto modCreateFunc = [&](mi::neuraylib::IMdl_execution_context* context)
    {
      return m_impExpApi->load_module_from_string(m_transaction.get(), moduleName.c_str(), srcStr.data(), context);
    };

    return compile(identifier, moduleName, modCreateFunc, compiledMaterial);
  }

  bool McMdlMaterialCompiler::compileFromFile(const char* filePath,
                                              std::string_view identifier,
                                              mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    std::string fileDir = fs::path(filePath).parent_path().string();
    std::string moduleName = GB_FMT("::{}", fs::path(filePath).stem().string());

    auto modCreateFunc = [&](mi::neuraylib::IMdl_execution_context* context)
    {
      McMdlEntityResolverUserData* userData = new McMdlEntityResolverUserData();
      userData->dirPrefix = fileDir;

      context->set_option("user_data", userData); // pass to entity resolver

      return m_impExpApi->load_module(m_transaction.get(), moduleName.c_str(), context);
    };

    return compile(identifier, moduleName, modCreateFunc, compiledMaterial);
  }

  bool McMdlMaterialCompiler::compile(std::string_view identifier,
                                      std::string_view moduleName,
                                      std::function<mi::Sint32(mi::neuraylib::IMdl_execution_context*)> modCreateFunc,
                                      mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_factory->create_execution_context());
    context->set_option("resolve_resources", false);

    mi::Sint32 modCreateResult = modCreateFunc(context.get());

    bool compResult = (modCreateResult == 0 || modCreateResult == 1) &&
      createCompiledMaterial(context.get(), moduleName.data(), identifier, compiledMaterial);

    m_logger->flushContextMessages(context.get());

    return compResult;
  }

  bool McMdlMaterialCompiler::createCompiledMaterial(mi::neuraylib::IMdl_execution_context* context,
                                                     std::string_view moduleName,
                                                     std::string_view identifier,
                                                     mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    mi::base::Handle<const mi::IString> moduleDbName(m_factory->get_db_module_name(moduleName.data()));
    assert(moduleDbName);
    mi::base::Handle<const mi::neuraylib::IModule> module(m_transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str()));
    assert(module);

    std::string materialDbName = GB_FMT("{}::{}", moduleDbName->get_c_str(), identifier);
    mi::base::Handle<const mi::IArray> funcs(module->get_function_overloads(materialDbName.c_str(), (const mi::neuraylib::IExpression_list*)nullptr));
    if (funcs->get_length() == 0)
    {
      std::string errorMsg = GB_FMT("material with identifier {} not found in MDL module", identifier);
      m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, errorMsg.c_str());
      return false;
    }
    if (funcs->get_length() > 1)
    {
      std::string errorMsg = GB_FMT("ambigious material identifier {} for MDL module", identifier);
      m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, errorMsg.c_str());
      return false;
    }

    mi::base::Handle<const mi::IString> exactMaterialDbName(funcs->get_element<mi::IString>(0));
    assert(exactMaterialDbName);

    mi::base::Handle<const mi::neuraylib::IFunction_definition> materialDefinition(m_transaction->access<mi::neuraylib::IFunction_definition>(exactMaterialDbName->get_c_str()));
    if (!materialDefinition)
    {
      return false;
    }

    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IFunction_call> materialInstance(materialDefinition->create_function_call(nullptr, &result));
    if (result != 0 || !materialInstance)
    {
      return false;
    }

    mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance2(materialInstance->get_interface<mi::neuraylib::IMaterial_instance>());
    if (!materialInstance2)
    {
      return false;
    }

    auto compileFlags = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS; // Instance compilation, no class compilation.
    compiledMaterial = mi::base::Handle<mi::neuraylib::ICompiled_material>(materialInstance2->create_compiled_material(compileFlags, context));

    return compiledMaterial != nullptr;
  }
}
