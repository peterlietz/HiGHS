// Unreported (found in a code review of latest)
// MIP with mip_lp_solver = ipm: after a restart, the root LP is not solved
// by an IPM, but by dual simplex from the root basis carried over into the
// restarted model
//
// Symptom on latest: only the first root LP goes to the IPM. On a large,
// dual degenerate time-space network MIP, the root LP after the restart ran
// over 175k dual simplex iterations without finishing, where IPX and
// crossover solve it in about 11 s. On the small models here, every root LP
// after a restart starts from a basis: bell5 restarts 7 times, lseu, rgn
// and issue-2173 once each.
//
// Cause: evaluateRootNode sets firstrootbasis (mapped into the restarted
// model by performRestart) on the LP before the root LP solve. With a valid
// basis, HighsLpRelaxation::run always uses simplex, whatever
// mip_lp_solver says.
//
// Fix: branch fixes/mip-restart-root-ipm (option mip_restart_root_ipm,
// default off: the carried-over basis is not set when mip_lp_solver is an
// IPM, so the restarted root LP is solved like the first one).
//
// Test design: the decisive assertion is about the behaviour, not the
// option. With log_dev_level >= 1 and the MIP timing bit of
// highs_analysis_level, HighsLpRelaxation::run logs "MIP-Timing: ... start
// first LP solve (with basis)" or "(without basis)" for the first LP solve
// after each (re)setup, that is, for each root LP. The test requires that
// every root LP starts without a basis, so that the IPM is used. Setting
// mip_restart_root_ipm fails on latest, which is ignored; a fix of the
// default upstream would pass the test as well.
#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

namespace {
HighsInt countOf(const std::string& text, const std::string& s) {
  HighsInt count = 0;
  for (size_t pos = text.find(s); pos != std::string::npos;
       pos = text.find(s, pos + 1))
    count++;
  return count;
}
}  // namespace

TEST_CASE("weakness-mip-restart-root-ipm-warm-start",
          "[known-issue][!shouldfail]") {
  for (const std::string model : {"bell5", "lseu", "rgn", "issue-2173"}) {
    Highs highs;
    LogCapture log;
    captureLog(highs, log);
    highs.setOptionValue("mip_lp_solver", kIpmString);
    // Unknown on latest: returns an error there, which is ignored
    highs.setOptionValue("mip_restart_root_ipm", true);
    // For the "MIP-Timing" lines
    highs.setOptionValue("log_dev_level", kHighsLogDevLevelInfo);
    highs.setOptionValue("highs_analysis_level", kHighsAnalysisLevelMipTime);
    REQUIRE(highs.readModel(instanceFile(model + ".mps")) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    const HighsInt num_restart =
        countOf(log.text, "inactive integer columns, restarting");
    const HighsInt num_root_lp_with_basis =
        countOf(log.text, "start first LP solve (with basis)");
    const HighsInt num_root_lp_without_basis =
        countOf(log.text, "start first LP solve (without basis)");
    INFO(model << ": " << num_restart << " restarts; root LPs with basis "
               << num_root_lp_with_basis << ", without basis "
               << num_root_lp_without_basis);
    // Setup: each of these models restarts
    REQUIRE(num_restart > 0);
    // latest: every root LP after a restart starts with the carried-over
    // basis, so it is solved by dual simplex
    REQUIRE(num_root_lp_with_basis == 0);
    REQUIRE(num_root_lp_without_basis == num_restart + 1);
    highs.resetGlobalScheduler(true);
  }
}
