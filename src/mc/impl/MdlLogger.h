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

#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wdeprecated-copy"
//#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-function"
#include <mi/base/interface_implement.h>
#include <mi/base/ilogger.h>
#include <mi/neuraylib/imdl_execution_context.h>
#pragma clang diagnostic pop

namespace gtl
{
  class McMdlLogger : public mi::base::Interface_implement<mi::base::ILogger>
  {
  public:
    void message(mi::base::Message_severity level,
                 const char* moduleCategory,
                 const mi::base::Message_details& details,
                 const char* message) override;

    void message(mi::base::Message_severity level,
                 const char* moduleCategory,
                 const char* message) override;

    void message(mi::base::Message_severity level,
                 const char* message);

    void flushContextMessages(mi::neuraylib::IMdl_execution_context* context);
  };
}
