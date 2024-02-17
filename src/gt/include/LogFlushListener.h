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

#pragma once

#include <doctest/doctest.h>

namespace gtl
{
  // This listener must be registered using the doctest
  // REGISTER_LISTENER(name, priority, class) macro.
  //
  // It prevents race conditions between gatling's logger
  // and doctest's iostream printing.
  struct GtLogFlushListener : doctest::IReporter
  {
    GtLogFlushListener(const doctest::ContextOptions&);

    void report_query(const doctest::QueryData&) override;

    void test_run_start() override;

    void test_run_end(const doctest::TestRunStats&) override;

    void test_case_start(const doctest::TestCaseData&) override;

    void test_case_reenter(const doctest::TestCaseData&) override;

    void test_case_end(const doctest::CurrentTestCaseStats&) override;

    void test_case_exception(const doctest::TestCaseException&) override;

    void subcase_start(const doctest::SubcaseSignature&) override;

    void subcase_end() override;

    void log_assert(const doctest::AssertData&) override;

    void log_message(const doctest::MessageData&) override;

    void test_case_skipped(const doctest::TestCaseData&) override;
  };
}
