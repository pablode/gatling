#include "MdlMaterialCompiler.h"

#include <mi/mdl_sdk.h>

#include <atomic>

namespace sg
{
  const char* MODULE_PREFIX = "::gatling_";
  std::atomic_uint32_t s_idCounter(0);

  std::string _makeModuleName(const std::string& identifier)
  {
    uint32_t uniqueId = ++s_idCounter;
    return std::string(MODULE_PREFIX) + std::to_string(uniqueId) + "_" + identifier;
  }

  MdlMaterialCompiler::MdlMaterialCompiler(MdlRuntime& runtime)
  {
    m_logger = mi::base::Handle<MdlLogger>(runtime.getLogger());
    m_database = mi::base::Handle<mi::neuraylib::IDatabase>(runtime.getDatabase());
    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(runtime.getTransaction());
    m_factory = mi::base::Handle<mi::neuraylib::IMdl_factory>(runtime.getFactory());
    m_impExpApi = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(runtime.getImpExpApi());
  }

  bool MdlMaterialCompiler::compileMaterial(const std::string& src,
                                            const std::string& identifier,
                                            mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_factory->create_execution_context());

    std::string moduleName = _makeModuleName(identifier);

    bool result = createModule(context.get(), moduleName.c_str(), src.c_str()) &&
                  createCompiledMaterial(context.get(), moduleName.c_str(), identifier, compiledMaterial);

    m_logger->flushContextMessages(context.get());

    return result;
  }

  bool MdlMaterialCompiler::createModule(mi::neuraylib::IMdl_execution_context* context,
                                         const char* moduleName,
                                         const char* mdlSrc)
  {
    mi::Sint32 result = m_impExpApi->load_module_from_string(m_transaction.get(), moduleName, mdlSrc, context);
    return result == 0 || result == 1;
  }

  bool MdlMaterialCompiler::createCompiledMaterial(mi::neuraylib::IMdl_execution_context* context,
                                                   const char* moduleName,
                                                   const std::string& identifier,
                                                   mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial)
  {
    mi::base::Handle<const mi::IString> moduleDbName(m_factory->get_db_module_name(moduleName));
    mi::base::Handle<const mi::neuraylib::IModule> module(m_transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str()));
    assert(module);

    std::string materialDbName = std::string(moduleDbName->get_c_str()) + "::" + identifier;
    mi::base::Handle<const mi::IArray> funcs(module->get_function_overloads(materialDbName.c_str(), (const mi::neuraylib::IExpression_list*)nullptr));
    if (funcs->get_length() == 0)
    {
      std::string errorMsg = std::string("Material with identifier ") + identifier + " not found in MDL module\n";
      m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, errorMsg.c_str());
      return false;
    }
    if (funcs->get_length() > 1)
    {
      std::string errorMsg = std::string("Ambigious material identifier ") + identifier + " for MDL module\n";
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
    return true;
  }
}
