#include "MdlHlslCodeGen.h"

#include <mi/mdl_sdk.h>

#include <sstream>
#include <cassert>

namespace sg
{
  const char* SCATTERING_FUNC_NAME = "mdl_bsdf_scattering";
  const char* EMISSION_FUNC_NAME = "mdl_edf_emission";
  const char* EMISSION_INTENSITY_FUNC_NAME = "mdl_edf_emission_intensity";
  const char* MATERIAL_STATE_NAME = "Shading_state_material";

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

  bool MdlHlslCodeGen::init(MdlRuntime& runtime)
  {
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi(runtime.getBackendApi());
    m_backend = mi::base::Handle<mi::neuraylib::IMdl_backend>(backendApi->get_backend(mi::neuraylib::IMdl_backend_api::MB_HLSL));
    if (!m_backend.is_valid_interface())
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "HLSL backend not supported by MDL runtime");
      return false;
    }

    m_logger = mi::base::Handle<MdlLogger>(runtime.getLogger());

    mi::base::Handle<mi::neuraylib::IMdl_factory> factory(runtime.getFactory());
    m_context = mi::base::Handle<mi::neuraylib::IMdl_execution_context>(factory->create_execution_context());

    m_database = mi::base::Handle<mi::neuraylib::IDatabase>(runtime.getDatabase());
    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(runtime.getTransaction());
    return true;
  }

  bool MdlHlslCodeGen::translate(const std::vector<const mi::neuraylib::ICompiled_material*>& materials,
                                 std::string& hlslSrc)
  {
    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(m_backend->create_link_unit(m_transaction.get(), m_context.get()));
    m_logger->flushContextMessages(m_context.get());

    if (!linkUnit)
    {
      return false;
    }

    uint32_t materialCount = materials.size();
    for (uint32_t i = 0; i < materialCount; i++)
    {
      const mi::neuraylib::ICompiled_material* material = materials.at(i);
      assert(material);

      if (!appendMaterialToLinkUnit(i, material, linkUnit.get()))
      {
        return false;
      }
    }

    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode(m_backend->translate_link_unit(linkUnit.get(), m_context.get()));
    m_logger->flushContextMessages(m_context.get());

    if (!targetCode)
    {
      return false;
    }

    if (targetCode->get_texture_count() > 0)
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, "Textures not supported, aborting\n");
      return false;
    }

    std::stringstream ss;
    ss << targetCode->get_code();

    _generateInOutSwitch(ss, SCATTERING_FUNC_NAME, "sample", "Bsdf_sample_data", materialCount);
    _generateInitSwitch(ss, SCATTERING_FUNC_NAME, materialCount);

    _generateInOutSwitch(ss, EMISSION_FUNC_NAME, "evaluate", "Edf_evaluate_data", materialCount);
    _generateInitSwitch(ss, EMISSION_FUNC_NAME, materialCount);

    _generateEdfIntensitySwitch(ss, materialCount);

    hlslSrc = ss.str();
    return true;
  }

  bool MdlHlslCodeGen::appendMaterialToLinkUnit(uint32_t idx,
                                                const mi::neuraylib::ICompiled_material* compiledMaterial,
                                                mi::neuraylib::ILink_unit* linkUnit)
  {
    std::string idxStr = std::to_string(idx);
    auto scatteringFuncName = std::string(SCATTERING_FUNC_NAME) + "_" + idxStr;
    auto emissionFuncName = std::string(EMISSION_FUNC_NAME) + "_" + idxStr;
    auto emissionIntensityFuncName = std::string(EMISSION_INTENSITY_FUNC_NAME) + "_" + idxStr;

    std::vector<mi::neuraylib::Target_function_description> genFunctions;
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.scattering", scatteringFuncName.c_str()));
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.emission.emission", emissionFuncName.c_str()));
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.emission.intensity", emissionIntensityFuncName.c_str()));

    mi::Sint32 result = linkUnit->add_material(
      compiledMaterial,
      genFunctions.data(),
      genFunctions.size(),
      m_context.get()
    );

    m_logger->flushContextMessages(m_context.get());

    return result == 0;
  }
}
