// ERGO-Code/HiGHS#3736 (open)
// HiPO: "NaN detected" one iteration from the optimum when the up-looking
// factorisation is used
//
// Symptom on latest: on this LP (19 equality rows, 83 columns, entries +-1,
// x >= 0, optimum 8), with presolve off, HiPO stops with "NaN detected"
// (Solve error) at iteration 10. The multifrontal factorisation, IPX and
// simplex all solve it.
//
// Cause: near the optimum A Theta A^T is numerically singular, and
// elimination cancels some pivots down to rounding noise.
// UpLookingSolver::factor regularises a pivot only below the absolute
// reg_threshold_ = 1e-13, so these pivots are used as computed, and the
// factor no longer represents the matrix.
//
// Fix: branch fixes/3736-hipo-uplooking-relreg, on top of
// fixes/3718-hipo-relative-pivot-reg, whose relative threshold it uses.
#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

#ifdef HIPO
using namespace known_issues;

TEST_CASE("issue-3736-hipo-uplooking-nan", "[known-issue][!shouldfail]") {
  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("presolve", kHighsOffString);
  REQUIRE(highs.readModel(dataFile("issue-3736.mps")) == HighsStatus::kOk);
  // latest: Solve error ("NaN detected")
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(highs.getInfo().objective_function_value == Approx(8));
}
#endif
