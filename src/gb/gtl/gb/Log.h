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

#include <quill/Logger.h>
#include <quill/LogMacros.h>
#include <quill/sinks/Sink.h>
#include <quill/std/Array.h>
#include <quill/std/FilesystemPath.h>
#include <quill/std/UnorderedMap.h>
#include <quill/std/UnorderedSet.h>
#include <quill/std/Vector.h>

#define GB_ERROR(fmt, ...) QUILL_LOG_ERROR(gtl::gbGetLogger(), fmt, ##__VA_ARGS__)
#define GB_ERROR_DYN(fmt, ...) QUILL_LOG_DYNAMIC(gtl::gbGetLogger(), quill::LogLevel::Error, fmt, ##__VA_ARGS__)
#define GB_WARN(fmt, ...) QUILL_LOG_WARNING(gtl::gbGetLogger(), fmt, ##__VA_ARGS__)
#define GB_LOG(fmt, ...) QUILL_LOG_INFO(gtl::gbGetLogger(), fmt, ##__VA_ARGS__)
#define GB_DEBUG(fmt, ...) QUILL_LOG_DEBUG(gtl::gbGetLogger(), fmt, ##__VA_ARGS__)
#define GB_DEBUG_DYN(fmt, ...) QUILL_LOG_DYNAMIC(gtl::gbGetLogger(), quill::LogLevel::Debug, fmt, ##__VA_ARGS__)

namespace gtl
{
  void gbLogInit(const std::vector<std::shared_ptr<quill::Sink>>& extraSinks = {});

  quill::Logger* gbGetLogger();

  void gbLogFlush();
}
