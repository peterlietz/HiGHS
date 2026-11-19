// Unreported (found in a code review of latest)
// MIP: when root separation stops for a restart, RENS and the feasibility
// pump still run on the model before the restart
//
// Symptom on latest: on a large time-space network MIP, RENS took 228-360 s
// on the root model before a restart that was certain, and ran again after
// it. On 2171 here, separation stops for a restart (50% of the integers
// are inactive), but RENS runs first: its dive finds the optimum, so the
// restart doesn't happen. With RENS off, the same run restarts at once.
//
// Cause: the separation loop of evaluateRootNode stops with stall = -1 when
// at least 10% of the integer columns are inactive (main MIP, presolve on).
// That percentage cannot fall, so with mip_allow_restart the restart test at
// the end of evaluateRootNode restarts unless the root node is solved
// before. RENS and the feasibility pump run before that test.
//
// Fix: branch fixes/mip-root-restart-first (option mip_root_restart_first,
// default off: RENS and the feasibility pump are skipped when a restart is
// pending, and the restart test is taken directly).
//
// Test design: the decisive assertion is about the behaviour, not the
// option. It compares two runs of 2171, with RENS on and off, and requires
// that RENS doesn't change anything up to the first display line after the
// first restart: if RENS doesn't run before a pending restart, the runs are
// identical up to there. With timeless_log and mip_min_logging_interval = 0,
// every display line is printed, so the display lines (seen by
// kCallbackMipLogging) don't depend on timing. Setting
// mip_root_restart_first fails on latest, which is ignored; a fix of the
// default upstream would pass the test as well.
#include <string>
#include <tuple>
#include <vector>

#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

namespace {
// Data of a MIP display line: nodes, LP iterations, primal and dual bound
using DisplayLine = std::tuple<int64_t, int64_t, double, double>;

struct RootTrace {
  std::string log;
  // Display lines up to the first one after the first restart
  std::vector<DisplayLine> display_line;
  bool restarted = false;
  bool done = false;
};

RootTrace traceToFirstRestart(const std::string& model, const bool rens) {
  Highs highs;
  RootTrace trace;
  highs.setOptionValue("output_flag", true);
  highs.setOptionValue("log_to_console", kDevRun);
  // Unknown on latest: returns an error there, which is ignored
  highs.setOptionValue("mip_root_restart_first", true);
  highs.setOptionValue("mip_heuristic_run_rens", rens);
  highs.setOptionValue("timeless_log", true);
  highs.setOptionValue("mip_min_logging_interval", 0.0);
  highs.setCallback(
      [](int callback_type, const std::string& message,
         const HighsCallbackOutput* data_out, HighsCallbackInput*,
         void* user_data) {
        RootTrace& trace = *static_cast<RootTrace*>(user_data);
        if (callback_type == kCallbackLogging) {
          trace.log += message;
          if (message.find("inactive integer columns, restarting") !=
              std::string::npos)
            trace.restarted = true;
        } else if (callback_type == kCallbackMipLogging && !trace.done) {
          trace.display_line.emplace_back(
              data_out->mip_node_count, data_out->mip_total_lp_iterations,
              data_out->mip_primal_bound, data_out->mip_dual_bound);
          if (trace.restarted) trace.done = true;
        }
      },
      &trace);
  highs.startCallback(kCallbackLogging);
  highs.startCallback(kCallbackMipLogging);
  REQUIRE(highs.readModel(instanceFile(model + ".mps")) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  highs.resetGlobalScheduler(true);
  return trace;
}
}  // namespace

TEST_CASE("weakness-mip-root-restart-first-rens-before-restart",
          "[known-issue][!shouldfail]") {
  const RootTrace with_rens = traceToFirstRestart("2171", true);
  const RootTrace without_rens = traceToFirstRestart("2171", false);
  // Setup: without RENS, root separation stops for a restart, which follows
  REQUIRE(without_rens.restarted);
  // Report the first display line that differs
  size_t k = 0;
  while (k < with_rens.display_line.size() &&
         k < without_rens.display_line.size() &&
         with_rens.display_line[k] == without_rens.display_line[k])
    k++;
  auto lineString = [](const std::vector<DisplayLine>& display_line,
                       const size_t k) {
    if (k >= display_line.size()) return std::string("none");
    return "nodes " + std::to_string(std::get<0>(display_line[k])) +
           ", LP iterations " + std::to_string(std::get<1>(display_line[k])) +
           ", primal bound " + std::to_string(std::get<2>(display_line[k])) +
           ", dual bound " + std::to_string(std::get<3>(display_line[k]));
  };
  INFO("with RENS: " << with_rens.display_line.size() << " display lines"
                     << (with_rens.restarted ? "" : " (no restart)")
                     << "; without RENS: " << without_rens.display_line.size()
                     << "; display line " << k << " with RENS: "
                     << lineString(with_rens.display_line, k)
                     << "; without RENS: "
                     << lineString(without_rens.display_line, k));
  // latest: RENS runs before the restart (its dive finds the optimum, so
  // the run doesn't restart)
  REQUIRE(with_rens.display_line == without_rens.display_line);
  REQUIRE(with_rens.restarted);
}
