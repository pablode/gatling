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
#include <gtl/gb/Log.h>

#include <mi/mdl_sdk.h>

#include <atomic>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

namespace
{
  using namespace gtl;

  std::atomic_uint32_t s_idCounter(0);

  std::string _MakeModuleName(std::string_view identifier)
  {
    uint32_t uniqueId = ++s_idCounter;
    return GB_FMT("::gatling::{}_{}", uniqueId, identifier);
  }

  mi::base::Handle<mi::neuraylib::IValue> _TranslateParameterValue(mi::neuraylib::IMdl_execution_context* context,
                                                                   mi::base::Handle<mi::neuraylib::ITransaction> transaction,
                                                                   const mi::base::Handle<mi::neuraylib::IMdl_factory>& mdlFactory,
                                                                   const mi::base::Handle<mi::neuraylib::IType_factory>& tf,
                                                                   const mi::base::Handle<mi::neuraylib::IValue_factory>& vf,
                                                                   const McMaterialParameterValue& value)
  {
    const auto makeVecValue = [&tf, &vf](const GbVec4f& compVals, uint32_t compCount)
    {
      mi::base::Handle<const mi::neuraylib::IType_float> floatType(tf->create_float());
      mi::base::Handle<const mi::neuraylib::IType_vector> vecType(tf->create_vector(floatType.get(), compCount));

      mi::base::Handle<mi::neuraylib::IValue_vector> vec(vf->create_vector(vecType.get()));
      vec->set_value(0, vf->create_float(compVals.x));
      vec->set_value(1, vf->create_float(compVals.y));
      if (compCount > 1)
      {
        vec->set_value(2, vf->create_float(compVals.z));
      }
      if (compCount > 2)
      {
        vec->set_value(3, vf->create_float(compVals.w));
      }

      return mi::base::Handle<mi::neuraylib::IValue>(vec);
    };

    if (std::holds_alternative<bool>(value))
    {
      return mi::base::Handle<mi::neuraylib::IValue>(vf->create_bool(std::get<bool>(value)));
    }
    else if (std::holds_alternative<int>(value))
    {
      return mi::base::Handle<mi::neuraylib::IValue>(vf->create_int(std::get<int>(value)));
    }
    else if (std::holds_alternative<float>(value))
    {
      return mi::base::Handle<mi::neuraylib::IValue>(vf->create_float(std::get<float>(value)));
    }
    else if (std::holds_alternative<GbVec2f>(value))
    {
      auto v = std::get<GbVec2f>(value);
      return makeVecValue(GbVec4f{ v.x, v.y, 0.0f, 0.0f }, 2);
    }
    else if (std::holds_alternative<GbVec3f>(value))
    {
      auto v = std::get<GbVec3f>(value);
      return makeVecValue(GbVec4f{ v.x, v.y, v.z, 0.0f }, 3);
    }
    else if (std::holds_alternative<GbVec4f>(value))
    {
      auto v = std::get<GbVec4f>(value);
      return makeVecValue(GbVec4f{ v.x, v.y, v.z, v.w }, 4);
    }
    else if (std::holds_alternative<GbColor>(value))
    {
      auto v = std::get<GbColor>(value);
      return mi::base::Handle<mi::neuraylib::IValue>(vf->create_color(v.r, v.g, v.b));
    }
    else if (std::holds_alternative<GbTextureAsset>(value))
    {
      auto texInfo = std::get<GbTextureAsset>(value);

      float gamma = texInfo.isSrgb ? 2.2f : 1.0f;
      mi::base::Handle<mi::neuraylib::IValue_texture> tex(mdlFactory->create_texture(
        transaction.get(), texInfo.absPath.c_str(), mi::neuraylib::IType_texture::TS_2D, gamma, nullptr, false, context
      ));

      return mi::base::Handle<mi::neuraylib::IValue>(tex);
    }
    else
    {
      GB_ERROR("coding error: unhandled material parameter type");
      return mi::base::Handle<mi::neuraylib::IValue>();
    }
  }
}

namespace gtl
{
  McMdlMaterialCompiler::McMdlMaterialCompiler(McMdlRuntime& runtime)
  {
    m_logger = runtime.getLogger();
    m_database = runtime.getDatabase();
    m_transaction = runtime.getTransaction();
    m_config = runtime.getConfig();
    m_factory = runtime.getFactory();
    m_impExpApi = runtime.getImpExpApi();
    m_vf = m_factory->create_value_factory(m_transaction.get());
    m_tf = m_factory->create_type_factory(m_transaction.get());
    m_ef = m_factory->create_expression_factory(m_transaction.get());
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
                                              mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial,
                                              const McMaterialParameters& params)
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

    return compile(identifier, moduleName, modCreateFunc, compiledMaterial, params);
  }

  bool McMdlMaterialCompiler::compile(std::string_view identifier,
                                      std::string_view moduleName,
                                      std::function<mi::Sint32(mi::neuraylib::IMdl_execution_context*)> modCreateFunc,
                                      mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial,
                                      const McMaterialParameters& params)
  {
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_factory->create_execution_context());
    context->set_option("resolve_resources", false);

    mi::Sint32 modCreateResult = modCreateFunc(context.get());

    bool compResult = (modCreateResult == 0 || modCreateResult == 1) &&
      createCompiledMaterial(context.get(), moduleName.data(), identifier, compiledMaterial, params);

    m_logger->flushContextMessages(context.get());

    return compResult;
  }

  bool McMdlMaterialCompiler::createCompiledMaterial(mi::neuraylib::IMdl_execution_context* context,
                                                     std::string_view moduleName,
                                                     std::string_view identifier,
                                                     mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial,
                                                     const McMaterialParameters& params)
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

    mi::neuraylib::IExpression_list* paramList = nullptr;
    if (!params.empty())
    {
      paramList = m_ef->create_expression_list();

      for (const auto& nameValuePair : params)
      {
        mi::base::Handle<mi::neuraylib::IValue> value = _TranslateParameterValue(context, m_transaction, m_factory, m_tf, m_vf, nameValuePair.second);

        if (!value)
        {
          continue;
        }

        mi::base::Handle<mi::neuraylib::IExpression> expr(m_ef->create_constant(value.get()));

        const std::string& name = nameValuePair.first;
        paramList->add_expression(name.c_str(), expr.get());
      }
    }

    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IFunction_call> materialInstance(materialDefinition->create_function_call(paramList, &result));
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
