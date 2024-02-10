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
