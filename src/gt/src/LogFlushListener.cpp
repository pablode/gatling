#include "LogFlushListener.h"

#include <Log.h>

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
