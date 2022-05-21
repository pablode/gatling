#include "MdlLogger.h"

#include <mi/mdl_sdk.h>

#include <stdio.h>

namespace sg
{
  const char* _miMessageSeverityToCStr(mi::base::Message_severity severity)
  {
    switch (severity)
    {
    case mi::base::MESSAGE_SEVERITY_FATAL:
      return "fatal";
    case mi::base::MESSAGE_SEVERITY_ERROR:
      return "error";
    case mi::base::MESSAGE_SEVERITY_WARNING:
      return "warning";
    case mi::base::MESSAGE_SEVERITY_INFO:
      return "info";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
      return "verbose";
    case mi::base::MESSAGE_SEVERITY_DEBUG:
      return "debug";
    default:
      break;
    }
    return "";
  }

  const char* _miMessageKindToCStr(mi::neuraylib::IMessage::Kind kind)
  {
    switch (kind)
    {
    case mi::neuraylib::IMessage::MSG_INTEGRATION:
      return "MDL SDK";
    case mi::neuraylib::IMessage::MSG_IMP_EXP:
      return "Importer/Exporter";
    case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
      return "Compiler Backend";
    case mi::neuraylib::IMessage::MSG_COMILER_CORE:
      return "Compiler Core";
    case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
      return "Compiler Archive Tool";
    case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
      return "Compiler DAG generator";
    default:
      break;
    }
    return "";
  }

  void MdlLogger::message(mi::base::Message_severity level,
                          const char* moduleCategory,
                          const mi::base::Message_details& details,
                          const char* message)
  {
#ifdef NDEBUG
    const mi::base::Message_severity minLogLevel = mi::base::MESSAGE_SEVERITY_ERROR;
#else
    const mi::base::Message_severity minLogLevel = mi::base::MESSAGE_SEVERITY_INFO;
#endif
    if (level <= minLogLevel)
    {
      const char* s_severity = _miMessageSeverityToCStr(level);
      FILE* os = (level <= mi::base::MESSAGE_SEVERITY_ERROR) ? stderr : stdout;
      fprintf(os, "[%s] (%s) %s\n", s_severity, moduleCategory, message);
#ifdef MI_PLATFORM_WINDOWS
      fflush(stderr);
#endif
    }
  }

  void MdlLogger::message(mi::base::Message_severity level,
                          const char* moduleCategory,
                          const char* message)
  {
    this->message(level, moduleCategory, mi::base::Message_details{}, message);
  }

  void MdlLogger::message(mi::base::Message_severity level,
                          const char* message)
  {
    const char* MODULE_CATEGORY = "shadergen";
    this->message(level, MODULE_CATEGORY, message);
  }

  void MdlLogger::flushContextMessages(mi::neuraylib::IMdl_execution_context* context)
  {
    for (mi::Size i = 0, n = context->get_messages_count(); i < n; ++i)
    {
      mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));

      const char* s_msg = message->get_string();
      const char* s_kind = _miMessageKindToCStr(message->get_kind());
      this->message(message->get_severity(), s_kind, s_msg);
    }
    context->clear_messages();
  }
}
