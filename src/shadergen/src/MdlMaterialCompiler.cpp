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

#include <mi/mdl_sdk.h>

#include <atomic>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

namespace sg
{
  const char* MODULE_PREFIX = "::gatling::";
  std::atomic_uint32_t s_idCounter(0);

  std::string _makeModuleName(std::string_view identifier)
  {
    uint32_t uniqueId = ++s_idCounter;
    return std::string(MODULE_PREFIX) + std::to_string(uniqueId) + "_" + std::string(identifier);
  }

  MdlMaterialCompiler::MdlMaterialCompiler(MdlRuntime& runtime, const std::string& mdlLibPath)
    : m_mdlLibPath(mdlLibPath)
  {
    m_logger = mi::base::Handle<MdlLogger>(runtime.getLogger());
    m_database = mi::base::Handle<mi::neuraylib::IDatabase>(runtime.getDatabase());
    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(runtime.getTransaction());
    m_config = mi::base::Handle<mi::neuraylib::IMdl_configuration>(runtime.getConfig());
    m_factory = mi::base::Handle<mi::neuraylib::IMdl_factory>(runtime.getFactory());
    m_impExpApi = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(runtime.getImpExpApi());
  }

  bool MdlMaterialCompiler::compileFromString(std::string_view srcStr,
                                              std::string_view identifier,
                                              mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    std::string moduleName = _makeModuleName(identifier);

    if (m_config->add_mdl_path(m_mdlLibPath.c_str()))
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "MaterialX MDL library files not found");
      return false;
    }

    auto modCreateFunc = [&](mi::neuraylib::IMdl_execution_context* context)
    {
      return m_impExpApi->load_module_from_string(m_transaction.get(), moduleName.c_str(), srcStr.data(), context);
    };

    bool result = compile(identifier, moduleName, modCreateFunc, compiledMaterial);

    m_config->remove_mdl_path(m_mdlLibPath.c_str());

    return result;
  }

  bool MdlMaterialCompiler::compileFromFile(std::string_view filePath,
                                            std::string_view identifier,
                                            mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    std::string fileDir = fs::path(filePath).parent_path().string();
    std::string moduleName = "::" + fs::path(filePath).stem().string();

    m_config->add_mdl_path(fileDir.c_str());
    // The free TurboSquid USD+MDL models, and possibly thousand paid ones too, come with some of the required Omni* files,
    // but some others are referenced and missing. If we include the directory of the asset as an MDL path after our own Omni*
    // MDL files, the Omni* files that come with the asset will be loaded instead of ours. They link to the other files that do
    // not exist, causing compilation to fail. By changing the load order, our complete Omni*-file suite will be used instead.
    if (m_config->add_mdl_path(m_mdlLibPath.c_str()))
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_WARNING, "MDL library files not found; code generation may fail");
    }

    auto modCreateFunc = [&](mi::neuraylib::IMdl_execution_context* context)
    {
      return m_impExpApi->load_module(m_transaction.get(), moduleName.c_str(), context);
    };

    bool result = compile(identifier, moduleName, modCreateFunc, compiledMaterial);

    m_config->remove_mdl_path(m_mdlLibPath.c_str());
    m_config->remove_mdl_path(fileDir.c_str());

    return result;
  }

  bool MdlMaterialCompiler::compile(std::string_view identifier,
                                    std::string_view moduleName,
                                    std::function<mi::Sint32(mi::neuraylib::IMdl_execution_context*)> modCreateFunc,
                                    mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_factory->create_execution_context());

    mi::Sint32 modCreateResult = modCreateFunc(context.get());

    bool compResult = (modCreateResult == 0 || modCreateResult == 1) &&
      createCompiledMaterial(context.get(), moduleName.data(), identifier, compiledMaterial);

    m_logger->flushContextMessages(context.get());

    return compResult;
  }

  bool MdlMaterialCompiler::createCompiledMaterial(mi::neuraylib::IMdl_execution_context* context,
                                                   std::string_view moduleName,
                                                   std::string_view identifier,
                                                   mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    mi::base::Handle<const mi::IString> moduleDbName(m_factory->get_db_module_name(moduleName.data()));
    mi::base::Handle<const mi::neuraylib::IModule> module(m_transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str()));
    assert(module);

    std::string materialDbName = std::string(moduleDbName->get_c_str()) + "::" + std::string(identifier);
    mi::base::Handle<const mi::IArray> funcs(module->get_function_overloads(materialDbName.c_str(), (const mi::neuraylib::IExpression_list*)nullptr));
    if (funcs->get_length() == 0)
    {
      std::string errorMsg = std::string("Material with identifier ") + std::string(identifier) + " not found in MDL module\n";
      m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, errorMsg.c_str());
      return false;
    }
    if (funcs->get_length() > 1)
    {
      std::string errorMsg = std::string("Ambigious material identifier ") + std::string(identifier) + " for MDL module\n";
      m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, errorMsg.c_str());
      return false;
    }

    mi::base::Handle<const mi::IString> exactMaterialDbName(funcs->get_element<mi::IString>(0));
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> matDefinition(m_transaction->access<mi::neuraylib::IMaterial_definition>(exactMaterialDbName->get_c_str()));
    if (!matDefinition)
    {
      return false;
    }

    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> matInstance(matDefinition->create_material_instance(NULL, &result));
    if (result != 0 || !matInstance)
    {
      return false;
    }

    auto flags = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS; // Instance compilation, no class compilation.
    compiledMaterial = mi::base::Handle<mi::neuraylib::ICompiled_material>(matInstance->create_compiled_material(flags, context));

    return compiledMaterial != nullptr;
  }
}
