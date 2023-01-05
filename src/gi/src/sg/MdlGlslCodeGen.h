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

#include <mi/base/handle.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_execution_context.h>

#include <stdint.h>
#include <string>
#include <vector>

#include "ShaderGen.h"
#include "MdlRuntime.h"
#include "MdlLogger.h"

namespace gi::sg
{
  class MdlGlslCodeGen
  {
  public:
    bool init(MdlRuntime& runtime);

    bool translate(const std::vector<const mi::neuraylib::ICompiled_material*>& materials,
                   std::string& glslSrc,
                   std::vector<TextureResource>& textureResources);

  private:
    void extractTextureInfos(mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode,
                             std::vector<TextureResource>& textureResources);

    bool appendMaterialToLinkUnit(uint32_t idx,
                                  const mi::neuraylib::ICompiled_material* compiledMaterial,
                                  mi::neuraylib::ILink_unit* linkUnit);

    std::string extractTargetCodeTextureFilePath(mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode, int i);

  private:
    mi::base::Handle<MdlLogger> m_logger;
    mi::base::Handle<mi::neuraylib::IMdl_backend> m_backend;
    mi::base::Handle<mi::neuraylib::IDatabase> m_database;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_context;
  };
}
