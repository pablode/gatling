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

#include "Log.h"

#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/sinks/ConsoleSink.h>

#include <assert.h>

namespace gtl
{
  static quill::Logger* s_logger = nullptr;

  void gbLogInit(const std::vector<std::shared_ptr<quill::Sink>>& extraSinks)
  {
    if (s_logger)
    {
      return;
    }

    quill::ConsoleSinkConfig::Colours consoleColors;
    consoleColors.apply_default_colours();
    consoleColors.assign_colour_to_log_level(quill::LogLevel::Info, quill::ConsoleSinkConfig::Colours::white);

    quill::ConsoleSinkConfig config;
    config.set_colours(consoleColors);

    auto sink = quill::Frontend::create_or_get_sink<quill::ConsoleSink>("console", config);

    std::vector<std::shared_ptr<quill::Sink>> sinks = extraSinks;
    sinks.push_back(std::move(sink));

    quill::PatternFormatterOptions formatOptions("[%(time)] (%(log_level)) %(message)", "%H:%M:%S.%Qms");
    s_logger = quill::Frontend::create_or_get_logger("root", sinks, formatOptions);
#ifdef GTL_VERBOSE
    s_logger->set_log_level(quill::LogLevel::Debug);
#endif

    quill::BackendOptions options;
    options.thread_name = "GbLog";
    quill::Backend::start(options);
  }

  quill::Logger* gbGetLogger()
  {
    return s_logger;
  }

  void gbLogFlush()
  {
    assert(s_logger);
    s_logger->flush_log();
  }
}
