//
// Copyright (C) 2024 Pablo Delgado Krämer
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

#include "LogFlushListener.h"

#include <gtl/gb/Log.h>

using namespace doctest;

namespace gtl
{
  // Unfortunately is it not enough to only hook report_query() or log_message().

  GtLogFlushListener::GtLogFlushListener(const ContextOptions&) {}

  void GtLogFlushListener::report_query(const QueryData&) { gbLogFlush(); }

  void GtLogFlushListener::test_run_start() { gbLogFlush(); }

  void GtLogFlushListener::test_run_end(const TestRunStats&) { gbLogFlush(); }

  void GtLogFlushListener::test_case_start(const TestCaseData&) { gbLogFlush(); }

  void GtLogFlushListener::test_case_reenter(const TestCaseData&) { gbLogFlush(); }

  void GtLogFlushListener::test_case_end(const CurrentTestCaseStats&) { gbLogFlush(); }

  void GtLogFlushListener::test_case_exception(const TestCaseException&) { gbLogFlush(); }

  void GtLogFlushListener::subcase_start(const SubcaseSignature&) { gbLogFlush(); }

  void GtLogFlushListener::subcase_end() { gbLogFlush(); }

  void GtLogFlushListener::log_assert(const AssertData&) { gbLogFlush(); }

  void GtLogFlushListener::log_message(const MessageData&) { gbLogFlush(); }

  void GtLogFlushListener::test_case_skipped(const TestCaseData&) { gbLogFlush(); }
}
