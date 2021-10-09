#include "MdlHlslTranslator.h"

#include <mi/mdl_sdk.h>

#include <sstream>
#include <cassert>

namespace sg
{
  const char* SCATTERING_FUNC_NAME = "mdl_bsdf_scattering";
  const char* EMISSION_FUNC_NAME = "mdl_edf_emission";
  const char* EMISSION_INTENSITY_FUNC_NAME = "mdl_edf_emission_intensity";
  const char* MATERIAL_STATE_NAME = "Shading_state_material";

  std::string _miMessageSeverityToString(mi::base::Message_severity severity)
  {
    switch (severity)
    {
    case mi::base::MESSAGE_SEVERITY_FATAL:
      return "fatal";
    case mi::base::MESSAGE_SEVERITY_ERROR:
      return "error";
    case mi::base::MESSAGE_SEVERITY_WARNING:
      return "warning";
    case mi::base::MESSAGE_SEVERITY_INFO:
      return "info";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
      return "verbose";
    case mi::base::MESSAGE_SEVERITY_DEBUG:
      return "debug";
    default:
      break;
    }
    return "";
  }

  std::string _miMessageKindToString(mi::neuraylib::IMessage::Kind kind)
  {
    switch (kind)
    {
    case mi::neuraylib::IMessage::MSG_INTEGRATION:
      return "MDL SDK";
    case mi::neuraylib::IMessage::MSG_IMP_EXP:
      return "Importer/Exporter";
    case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
      return "Compiler Backend";
    case mi::neuraylib::IMessage::MSG_COMILER_CORE:
      return "Compiler Core";
    case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
      return "Compiler Archive Tool";
    case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
      return "Compiler DAG generator";
    default:
      break;
    }
    return "";
  }

  void _generateInitSwitch(std::stringstream& ss,
                           const char* funcName,
                           uint32_t caseCount)
  {
    ss << "void " << funcName << "_init(in int idx, in " << MATERIAL_STATE_NAME << " sIn)\n";
    ss << "{\n";
    ss << "\tswitch(idx)\n";
    ss << "\t{\n";
    for (uint32_t i = 0; i < caseCount; i++)
    {
      ss << "\t\tcase " << i << ": " << funcName << "_" << i << "_init" << "(sIn); return;\n";
    }
    ss << "\t}\n";
    ss << "}\n";
  }

  void _generateEdfIntensitySwitch(std::stringstream& ss,
                                   uint32_t caseCount)
  {
    ss << "float3 " << EMISSION_INTENSITY_FUNC_NAME << "(in int idx, in " << MATERIAL_STATE_NAME << " sIn)\n";
    ss << "{\n";
    ss << "\tswitch(idx)\n";
    ss << "\t{\n";
    for (uint32_t i = 0; i < caseCount; i++)
    {
      ss << "\t\tcase " << i << ": return " << EMISSION_INTENSITY_FUNC_NAME << "_" << i << "(sIn);\n";
    }
    ss << "\t}\n";
    ss << "\treturn float3(0.0, 0.0, 0.0);\n";
    ss << "}\n";
  }

  void _generateInOutSwitch(std::stringstream& ss,
                             const char* funcName,
                             const char* opName,
                             const char* inoutTypeName,
                             uint32_t caseCount)
  {

    ss << "void " << funcName << "_" << opName << "(in int idx, inout " << inoutTypeName << " sInOut, in " << MATERIAL_STATE_NAME << " sIn)\n";
    ss << "{\n";
    ss << "\tswitch(idx)\n";
    ss << "\t{\n";
    for (uint32_t i = 0; i < caseCount; i++)
    {
      ss << "\t\tcase " << i << ": "  << funcName << "_" << i << "_" << opName << "(sInOut, sIn); return;\n";
    }
    ss << "\t}\n";
    ss << "}\n";
  }

  class MdlLogger : public mi::base::Interface_implement<mi::base::ILogger>
  {
  public:
    void message(mi::base::Message_severity level,
                 const char* moduleCategory,
                 const mi::base::Message_details& details,
                 const char* message) override
    {
#ifdef NDEBUG
      const mi::base::Message_severity minLogLevel = mi::base::MESSAGE_SEVERITY_WARNING;
#else
      const mi::base::Message_severity minLogLevel = mi::base::MESSAGE_SEVERITY_DEBUG;
#endif
      if (level <= minLogLevel)
      {
        const std::string s_severity = _miMessageSeverityToString(level);
        FILE* os = (level <= mi::base::MESSAGE_SEVERITY_ERROR) ? stderr : stdout;
        fprintf(os, "[%s] (%s) %s\n", s_severity.c_str(), moduleCategory, message);
#ifdef MI_PLATFORM_WINDOWS
        fflush(stderr);
#endif
      }
    }
  };

  const char* MODULE_PREFIX = "::gatling_";
  const char* MODULE_CATEGORY = "shadergen";

  bool MdlHlslTranslator::init(mi::neuraylib::INeuray& neuray,
                               const char* mtlxmdlPath)
  {
    m_logger = mi::base::make_handle(new MdlLogger());

    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdlConfig(neuray.get_api_component<mi::neuraylib::IMdl_configuration>());
    mdlConfig->set_logger(m_logger.get());
    mdlConfig->add_mdl_user_paths();

    if (mdlConfig->add_mdl_path(mtlxmdlPath))
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, MODULE_CATEGORY, "MaterialX MDL file path not found, translation not possible");
      return false;
    }

    m_mdlFactory = mi::base::Handle<mi::neuraylib::IMdl_factory>(neuray.get_api_component<mi::neuraylib::IMdl_factory>());
    m_mdlContext = mi::base::Handle<mi::neuraylib::IMdl_execution_context>(m_mdlFactory->create_execution_context());

    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdlBackendApi(neuray.get_api_component<mi::neuraylib::IMdl_backend_api>());
    m_mdlBackend = mdlBackendApi->get_backend(mi::neuraylib::IMdl_backend_api::MB_HLSL);
    if (!m_mdlBackend.is_valid_interface())
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, MODULE_CATEGORY, "HLSL backend not supported by MDL runtime");
      return false;
    }

    m_mdlImpExpApi = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(neuray.get_api_component<mi::neuraylib::IMdl_impexp_api>());
    m_mdlDatabase = mi::base::Handle<mi::neuraylib::IDatabase>(neuray.get_api_component<mi::neuraylib::IDatabase>());

    printContextMessages();
    return true;
  }

  bool MdlHlslTranslator::translate(const std::vector<const SourceIdentifierPair*>& input,
                                    std::string& hlslSrc)
  {
    mi::base::Handle<mi::neuraylib::IScope> scope(m_mdlDatabase->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(m_mdlBackend->create_link_unit(transaction.get(), m_mdlContext.get()));
    if (!linkUnit)
    {
      return false;
    }

    uint32_t moduleCount = input.size();
    for (uint32_t i = 0; i < input.size(); i++)
    {
      const SourceIdentifierPair& pair = *input.at(i);

      if (!appendModuleToLinkUnit(pair, i, transaction.get(), linkUnit.get()))
      {
        transaction->abort();
        return false;
      }
    }
    transaction->commit();

    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode(m_mdlBackend->translate_link_unit(linkUnit.get(), m_mdlContext.get()));
    printContextMessages();

    if (!targetCode)
    {
      return false;
    }

    std::stringstream ss;
    ss << targetCode->get_code();

    _generateInOutSwitch(ss, SCATTERING_FUNC_NAME, "sample", "Bsdf_sample_data", moduleCount);
    _generateInitSwitch(ss, SCATTERING_FUNC_NAME, moduleCount);

    _generateInOutSwitch(ss, EMISSION_FUNC_NAME, "evaluate", "Edf_evaluate_data", moduleCount);
    _generateInitSwitch(ss, EMISSION_FUNC_NAME, moduleCount);

    _generateEdfIntensitySwitch(ss, moduleCount);

    hlslSrc = ss.str();
    return true;
  }

  bool MdlHlslTranslator::appendModuleToLinkUnit(const SourceIdentifierPair& sourceAndIdentifier,
                                                 uint32_t idx,
                                                 mi::neuraylib::ITransaction* transaction,
                                                 mi::neuraylib::ILink_unit* linkUnit)
  {
    std::string moduleName = std::string(MODULE_PREFIX) + std::to_string(idx);

    mi::Sint32 result = m_mdlImpExpApi->load_module_from_string(transaction,
                                                                moduleName.c_str(),
                                                                sourceAndIdentifier.src.c_str(),
                                                                m_mdlContext.get());
    printContextMessages();

    if (result != 0 && result != 1)
    {
      return false;
    }

    mi::base::Handle<const mi::IString> moduleDbName(m_mdlFactory->get_db_module_name(moduleName.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str()));
    assert(module);

    const std::string& identifier = sourceAndIdentifier.identifier;
    std::string materialDbName = std::string(moduleDbName->get_c_str()) + "::" + identifier;
    mi::base::Handle<const mi::IArray> funcs(module->get_function_overloads(materialDbName.c_str(), (const mi::neuraylib::IExpression_list*)nullptr));
    if (funcs->get_length() == 0)
    {
      std::string errorMsg = std::string("Material with identifier ") + identifier + " not found in MDL module\n";
      m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, MODULE_CATEGORY, errorMsg.c_str());
      return false;
    }
    if (funcs->get_length() > 1)
    {
      std::string errorMsg = std::string("Ambigious material identifier ") + identifier + " for MDL module\n";
      m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, MODULE_CATEGORY, errorMsg.c_str());
      return false;
    }

    mi::base::Handle<const mi::IString> exactMaterialDbName(funcs->get_element<mi::IString>(0));
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> matDefinition(transaction->access<mi::neuraylib::IMaterial_definition>(exactMaterialDbName->get_c_str()));
    printContextMessages();

    if (!matDefinition)
    {
      return false;
    }

    mi::base::Handle<mi::neuraylib::IMaterial_instance> matInstance(matDefinition->create_material_instance(NULL, &result));
    if (result != 0 || !matInstance)
    {
      return false;
    }

    auto flags = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS; // Instance compilation, no class compilation.
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial(matInstance->create_compiled_material(flags, m_mdlContext.get()));
    printContextMessages();

    if (!compiledMaterial)
    {
      return false;
    }

    std::string idxStr = std::to_string(idx);
    auto scatteringFuncName = std::string(SCATTERING_FUNC_NAME) + "_" + idxStr;
    auto emissionFuncName = std::string(EMISSION_FUNC_NAME) + "_" + idxStr;
    auto emissionIntensityFuncName = std::string(EMISSION_INTENSITY_FUNC_NAME) + "_" + idxStr;

    std::vector<mi::neuraylib::Target_function_description> genFunctions;
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.scattering", scatteringFuncName.c_str()));
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.emission.emission", emissionFuncName.c_str()));
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.emission.intensity", emissionIntensityFuncName.c_str()));

    result = linkUnit->add_material(
      compiledMaterial.get(),
      genFunctions.data(),
      genFunctions.size(),
      m_mdlContext.get()
    );
    printContextMessages();

    return result == 0;
  }

  void MdlHlslTranslator::printContextMessages()
  {
    for (mi::Size i = 0, n = m_mdlContext->get_messages_count(); i < n; ++i)
    {
      const mi::base::Handle<const mi::neuraylib::IMessage> message(m_mdlContext->get_message(i));

      const char* s_msg = message->get_string();
      const std::string s_kind = _miMessageKindToString(message->get_kind());
      m_logger->message(message->get_severity(), s_kind.c_str(), s_msg);
    }
    m_mdlContext->clear_messages();
  }
}
