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

#include "MdlGlslCodeGen.h"

#include <mi/mdl_sdk.h>

#include <sstream>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

namespace gi::sg
{
  const char* SCATTERING_FUNC_NAME = "mdl_bsdf_scattering";
  const char* EMISSION_FUNC_NAME = "mdl_edf_emission";
  const char* EMISSION_INTENSITY_FUNC_NAME = "mdl_edf_emission_intensity";
  const char* THIN_WALLED_FUNC_NAME = "mdl_thin_walled";
  const char* VOLUME_ABSORPTION_FUNC_NAME = "mdl_absorption_coefficient";
  const char* CUTOUT_OPACITY_FUNC_NAME = "mdl_cutout_opacity";
  const char* MATERIAL_STATE_NAME = "State";

  bool MdlGlslCodeGen::init(MdlRuntime& runtime)
  {
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi(runtime.getBackendApi());
    m_backend = mi::base::Handle<mi::neuraylib::IMdl_backend>(backendApi->get_backend(mi::neuraylib::IMdl_backend_api::MB_GLSL));
    if (!m_backend.is_valid_interface())
    {
      m_logger->message(mi::base::MESSAGE_SEVERITY_FATAL, "GLSL backend not supported by MDL runtime");
      return false;
    }

    m_backend->set_option("enable_exceptions", "off");
    m_backend->set_option("use_renderer_adapt_normal", "on");

    m_logger = mi::base::Handle<MdlLogger>(runtime.getLogger());

    mi::base::Handle<mi::neuraylib::IMdl_factory> factory(runtime.getFactory());
    m_context = mi::base::Handle<mi::neuraylib::IMdl_execution_context>(factory->create_execution_context());
    m_context->set_option("resolve_resources", false);

    m_database = mi::base::Handle<mi::neuraylib::IDatabase>(runtime.getDatabase());
    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(runtime.getTransaction());
    return true;
  }

  bool MdlGlslCodeGen::genMaterialShadingCode(const mi::neuraylib::ICompiled_material* material,
                                              MdlGlslCodeGenResult& result)
  {
    std::vector<mi::neuraylib::Target_function_description> genFunctions;
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.scattering", SCATTERING_FUNC_NAME));
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.emission.emission", EMISSION_FUNC_NAME));
    genFunctions.push_back(mi::neuraylib::Target_function_description("surface.emission.intensity", EMISSION_INTENSITY_FUNC_NAME));
    genFunctions.push_back(mi::neuraylib::Target_function_description("thin_walled", THIN_WALLED_FUNC_NAME));
    genFunctions.push_back(mi::neuraylib::Target_function_description("volume.absorption_coefficient", VOLUME_ABSORPTION_FUNC_NAME));

    return generateGlslWithDfs(material, genFunctions, result.glslSource, result.textureResources);
  }

  bool MdlGlslCodeGen::genMaterialOpacityCode(const mi::neuraylib::ICompiled_material* material,
                                              MdlGlslCodeGenResult& result)
  {
    std::vector<mi::neuraylib::Target_function_description> genFunctions;
    genFunctions.push_back(mi::neuraylib::Target_function_description("geometry.cutout_opacity", CUTOUT_OPACITY_FUNC_NAME));

    return generateGlslWithDfs(material, genFunctions, result.glslSource, result.textureResources);
  }

  bool MdlGlslCodeGen::generateGlslWithDfs(const mi::neuraylib::ICompiled_material* compiledMaterial,
                                           std::vector<mi::neuraylib::Target_function_description>& genFunctions,
                                           std::string& glslSrc,
                                           std::vector<TextureResource>& textureResources)
  {
    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(m_backend->create_link_unit(m_transaction.get(), m_context.get()));
    m_logger->flushContextMessages(m_context.get());

    if (!linkUnit)
    {
      return false;
    }

    mi::Sint32 linkResult = linkUnit->add_material(
      compiledMaterial,
      genFunctions.data(),
      genFunctions.size(),
      m_context.get()
    );
    m_logger->flushContextMessages(m_context.get());

    if (linkResult)
    {
      return false;
    }

    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode(m_backend->translate_link_unit(linkUnit.get(), m_context.get()));
    m_logger->flushContextMessages(m_context.get());

    if (!targetCode)
    {
      return false;
    }

    assert(targetCode->get_ro_data_segment_count() == 0);

    extractTextureInfos(targetCode, textureResources);
    glslSrc = targetCode->get_code();

    return true;
  }

  void MdlGlslCodeGen::extractTextureInfos(mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode,
                                           std::vector<TextureResource>& textureResources)
  {
    size_t texCount = targetCode->get_body_texture_count();
    textureResources.reserve(texCount);

    // We start at 1 because index 0 is the invalid texture.
    for (int i = 1; i < texCount; i++)
    {
      TextureResource textureResource;
      textureResource.binding = i - 1;
      textureResource.is3dImage = false;
      textureResource.width = 1;
      textureResource.height = 1;
      textureResource.depth = 1;
      textureResource.data.resize(4, 0); // fall back to 1x1 black pixel

      switch (targetCode->get_texture_shape(i))
      {
      case mi::neuraylib::ITarget_code::Texture_shape_2d: {
        std::string filePath = extractTargetCodeTextureFilePath(targetCode, i);
        if (filePath.empty())
        {
          m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, "2d texture has no URL");
          break;
        }
        textureResource.filePath = filePath;
        break;
      }
      case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data: {
        mi::Size width, height, depth;
        const mi::Float32* dataPtr = targetCode->get_texture_df_data(i, width, height, depth);
        assert(dataPtr);

        textureResource.is3dImage = true;
        textureResource.width = (uint32_t) width;
        textureResource.height = (uint32_t) height;
        textureResource.depth = (uint32_t) depth;

        uint64_t size = width * height * depth * sizeof(mi::Float32);
        std::vector<uint8_t>& data = textureResource.data;
        data.resize(size);
        memcpy(&data[0], dataPtr, data.size());
        break;
      }
      case mi::neuraylib::ITarget_code::Texture_shape_3d:
        m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, "3d textures not supported");
        break;
      case mi::neuraylib::ITarget_code::Texture_shape_cube:
        m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, "Cube maps not supported");
        break;
      case mi::neuraylib::ITarget_code::Texture_shape_ptex:
        m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, "Ptex textures not supported");
        break;
      case mi::neuraylib::ITarget_code::Texture_shape_invalid:
        m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR, "Unknown texture type");
        break;
      default:
        assert(false);
        break;
      }

      textureResources.push_back(textureResource);
    }
  }

  std::string MdlGlslCodeGen::extractTargetCodeTextureFilePath(mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode, int i)
  {
    const char* url = targetCode->get_texture_url(i);
    if (!url)
    {
      return "";
    }

    std::string path(url);

    // If the MDL code is not generated but from a file, we need to convert relative to absolute file paths.
    const char* ownerModule = targetCode->get_texture_owner_module(i);
    if (ownerModule && strlen(ownerModule) > 0)
    {
      auto moduleDbName = std::string("mdl") + ownerModule;

      mi::base::Handle<const mi::neuraylib::IModule> module(m_transaction->access<const mi::neuraylib::IModule>(moduleDbName.c_str()));

      if (module)
      {
        fs::path parentPath = fs::path(module->get_filename()).parent_path();
        path = (parentPath / path).string();
      }
    }

#if _WIN32
    // MDL paths start with '/c/', but we need 'c:/' on Windows.
    bool shouldHaveDriveSpecifier = path.size() > 2 && path[0] == '/' && path[2] == '/';

    if (shouldHaveDriveSpecifier)
    {
      path[0] = path[1];
      path[1] = ':';
    }
#endif

    return path;
  }
}
