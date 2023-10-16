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

#include "log.h"

namespace gtl
{
  void gbLogInit()
  {
    std::shared_ptr<quill::Handler> consoleHandler = quill::stdout_handler();

    const char* logPattern = "[%(ascii_time)] (%(level_name)) %(message)";
    const char* timestampFormat = "%H:%M:%S.%Qms";
    consoleHandler->set_pattern(logPattern, timestampFormat);

    quill::Config cfg;
    cfg.default_handlers.emplace_back(consoleHandler);
    quill::configure(cfg);

    quill::start();
  }
}
