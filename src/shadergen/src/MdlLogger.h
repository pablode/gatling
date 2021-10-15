#pragma once

#include <mi/base/interface_implement.h>
#include <mi/base/ilogger.h>
#include <mi/neuraylib/imdl_execution_context.h>

namespace sg
{
  class MdlLogger : public mi::base::Interface_implement<mi::base::ILogger>
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
