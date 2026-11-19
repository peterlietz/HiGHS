// Helpers for the reproducers of open upstream issues in check/known-issues
#ifndef CHECK_KNOWN_ISSUES_H_
#define CHECK_KNOWN_ISSUES_H_

#include <string>

#include "HCheckConfig.h"
#include "Highs.h"

namespace known_issues {

// Set to true to see the solver logs of the known-issue tests
const bool kDevRun = false;

inline std::string dataFile(const std::string& name) {
  return std::string(HIGHS_DIR) + "/check/known-issues/data/" + name;
}

inline std::string instanceFile(const std::string& name) {
  return std::string(HIGHS_DIR) + "/check/instances/" + name;
}

// Collects the HiGHS log of a solve, so that a test can look for messages,
// without writing it to the console unless kDevRun is set
struct LogCapture {
  std::string text;
  bool contains(const std::string& s) const {
    return text.find(s) != std::string::npos;
  }
};

inline void captureLog(Highs& highs, LogCapture& log) {
  highs.setOptionValue("output_flag", true);
  highs.setOptionValue("log_to_console", kDevRun);
  highs.setCallback(
      [](int callback_type, const std::string& message,
         const HighsCallbackOutput*, HighsCallbackInput*, void* user_data) {
        if (callback_type == kCallbackLogging)
          static_cast<LogCapture*>(user_data)->text += message;
      },
      &log);
  highs.startCallback(kCallbackLogging);
}

}  // namespace known_issues

#endif
