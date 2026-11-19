// ERGO-Code/HiGHS#3718 (open)
// HiPO stalls at a P-D gap of about 1e-5 on degenerate LPs: tiny pivots
// from cancellation in the normal equations are not regularised
//
// Symptom on latest: on the LP relaxation of check/instances/3015.mps,
// HiPO's gap stops falling at about 5.7e-6, the stagnation test fires
// ("HiPO stagnated but HiGHS considers the solution acceptable"), and HiGHS
// reports "Optimal" at 9.37 times the tolerance (P-D objective error
// 8.4e-7).
//
// Cause: near the optimum of a primal-degenerate LP, elimination in
// A Theta A^T cancels pivots from their diagonal entry (about 1e10) down to
// rounding noise. The dense kernel only regularises pivots below
// M_norm1 * kDynamicDiagCoeff, an absolute threshold, so noise pivots of
// either sign are used, and the direction gets huge spurious components.
//
// Fix: branch fixes/3718-hipo-relative-pivot-reg (regularise pivots
// relative to their own diagonal entry).
#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

#ifdef HIPO
using namespace known_issues;

TEST_CASE("issue-3718-hipo-cancelled-pivots-stall",
          "[known-issue][!shouldfail]") {
  Highs highs;
  LogCapture log;
  captureLog(highs, log);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("solve_relaxation", true);
  REQUIRE(highs.readModel(instanceFile("3015.mps")) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  // latest: HiPO stagnates
  REQUIRE_FALSE(log.contains("HiPO stagnated"));
  double optimality_tolerance;
  highs.getOptionValue("optimality_tolerance", optimality_tolerance);
  // latest: 8.4e-7
  REQUIRE(highs.getInfo().primal_dual_objective_error <=
          optimality_tolerance);
}
#endif
