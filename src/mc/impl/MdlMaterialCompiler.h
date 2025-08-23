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

#pragma once

#include <string_view>
#include <functional>
#include <vector>

#include <mi/base/handle.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_impexp_api.h>

#include "MaterialParameters.h"
#include "MdlLogger.h"

namespace gtl
{
  class McMdlRuntime;

  class McMdlMaterialCompiler
  {
  public:
    McMdlMaterialCompiler(McMdlRuntime& runtime);

  public:
    bool compileFromString(std::string_view srcStr,
                           std::string_view identifier,
                           mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial);

    bool compileFromFile(const char* filePath,
                         std::string_view identifier,
                         mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial,
                         const McMaterialParameters& params = {});

  private:
    bool compile(std::string_view identifier,
                 std::string_view moduleName,
                 std::function<mi::Sint32(mi::neuraylib::IMdl_execution_context*)> modCreateFunc,
                 mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial,
                 const McMaterialParameters& params = {});

    bool createCompiledMaterial(mi::neuraylib::IMdl_execution_context* context,
                                std::string_view moduleName,
                                std::string_view identifier,
                                mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial,
                                const McMaterialParameters& params = {});

  private:
    mi::base::Handle<McMdlLogger> m_logger;
    mi::base::Handle<mi::neuraylib::IDatabase> m_database;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::neuraylib::IMdl_configuration> m_config;
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_factory;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_impExpApi;
    mi::base::Handle<mi::neuraylib::IValue_factory> m_vf;
    mi::base::Handle<mi::neuraylib::IType_factory> m_tf;
    mi::base::Handle<mi::neuraylib::IExpression_factory> m_ef;
  };
}
