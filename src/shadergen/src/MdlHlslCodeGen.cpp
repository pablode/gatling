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
                                 std::string& hlslSrc,
                                 std::vector<TextureResource>& textureResources)
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

    assert(targetCode->get_ro_data_segment_count() == 0);

    int texCount = targetCode->get_body_texture_count();
    textureResources.reserve(texCount);

    // Index 0 always is the invalid texture.
    for (int i = 1; i < texCount; i++)
    {
      const char* texDbName = targetCode->get_texture(i);
      assert(texDbName);
      mi::base::Handle<const mi::neuraylib::ITexture> texture(m_transaction->access<mi::neuraylib::ITexture>(texDbName));
      mi::base::Handle<const mi::neuraylib::IImage> image(m_transaction->access<mi::neuraylib::IImage>(texture->get_image()));

      TextureResource textureResource;
      textureResource.binding = i;

      uint32_t frameId = 0;
      uint32_t uvTileId = 0;
      uint32_t level = 0;

      switch (targetCode->get_texture_shape(i))
      {
      case mi::neuraylib::ITarget_code::Texture_shape_2d: {
        textureResource.width = image->resolution_x(frameId, uvTileId, level);
        textureResource.height = image->resolution_y(frameId, uvTileId, level);
        textureResource.filePath = image->get_filename(frameId, uvTileId);
        break;
      }
      case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data: {
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(frameId, uvTileId, level));
        assert(canvas);

        textureResource.width = canvas->get_resolution_x();
        textureResource.height = canvas->get_resolution_y();
        assert(canvas->get_layers_size() == 1);

        size_t size = textureResource.width * textureResource.height * 4;
        mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0));

        std::vector<uint8_t>& data = textureResource.data;
        data.resize(size);
        memcpy(&data[0], tile->get_data(), data.size());
        break;
      }
      case mi::neuraylib::ITarget_code::Texture_shape_3d:
        m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "3d textures not supported");
        return false;
      case mi::neuraylib::ITarget_code::Texture_shape_cube:
        m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "Cube maps not supported");
        return false;
      case mi::neuraylib::ITarget_code::Texture_shape_ptex:
        m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "Ptex textures not supported");
        return false;
      case mi::neuraylib::ITarget_code::Texture_shape_invalid:
      default:
        m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "Unknown texture type");
        assert(false);
        return false;
      }

      textureResources.push_back(textureResource);
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
