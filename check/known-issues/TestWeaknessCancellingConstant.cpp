// Unreported (found when analysing the dominant-offset weakness)
// IPX and HiPDLP don't terminate on a small LP whose objective constant
// cancels c'x: they run until the iteration or time limit
//
// The LP: a balanced transportation LP with 5 supplies of about 4e7, 8
// demands and 40 columns; the optimal c'x is 59861304428 and the objective
// constant is -59861304428, so the optimum is 0. Its names are generic, it
// was generated, not taken from a private model.
//
// IPX (Iterate::optimal) normalises the gap by 1 + |(p+d)/2 - fixed cost|
// and HiPDLP (relative_obj_gap) by 1 + |p| + |d|, where p and d include the
// constant, so both are about 1 here. Their stop tests then need |p - d|
// below 1e-8 and 1e-7, while the terms of p and d are about 6e10, so their
// rounding error alone is about 1e-5. On latest:
// - IPX, after HiGHS presolve has removed one dependent equation, reaches
//   |p - d| = 2.16e-6 at iteration 11 and stays there, with complementarity
//   about 3e-22. Its steps are 1.00 (primal) and 0.31 (dual), which none of
//   its bad-iteration rules counts, so its "no progress" exit never fires.
//   Simplex cleanup can't start, because IPX doesn't stop.
// - HiPDLP has no exit for a stalled run.
// Simplex (14 iterations), HiPO and cuPDLP solve the LP; HiPO and HiGHS's
// final check normalise the gap without the constant.
// On 3f617408f0, an earlier latest, IPX stops with "no progress" after 9
// iterations and simplex cleans up, and HiPDLP is Optimal.
//
// Symptom on latest (1916a10d9d), default options, with the iteration
// limits below (both are unlimited by default): IPX 1000 iterations,
// HiPDLP 1000000 iterations, both "Iteration limit reached". From the
// command line without a limit, both run until the time limit.
//
// Fixes:
// - fixes/dominant-offset-ipx-pdlp (stacked on
//   fixes/3715-hipo-pd-objective-error) leaves the constant out of the
//   normalisers. IPX is then Optimal after 8 iterations (9 with crossover
//   on), HiPDLP after 360.
// - fixes/ipx-hipdlp-stall-exit makes both stop when they stall: IPX after
//   16 iterations with "no progress", after which simplex cleans up or
//   HiGHS's final check accepts the point, so HiGHS reports Optimal;
//   HiPDLP after 154880 iterations with Unknown. So the HiPDLP case needs
//   the first fix to pass.
#include <string>

#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

TEST_CASE("weakness-cancelling-constant-ipx", "[known-issue][!shouldfail]") {
  for (const std::string crossover : {kHighsOnString, kHighsOffString}) {
    Highs highs;
    highs.setOptionValue("output_flag", kDevRun);
    highs.setOptionValue("solver", kIpxString);
    highs.setOptionValue("run_crossover", crossover);
    highs.setOptionValue("ipm_iteration_limit", 1000);
    REQUIRE(highs.readModel(dataFile("weakness-cancelling-constant.mps")) ==
            HighsStatus::kOk);
    highs.run();
    // latest: Iteration limit reached after 1000 IPM iterations
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double optimality_tolerance;
    highs.getOptionValue("optimality_tolerance", optimality_tolerance);
    REQUIRE(highs.getInfo().primal_dual_objective_error <=
            optimality_tolerance);
  }
}

TEST_CASE("weakness-cancelling-constant-hipdlp",
          "[known-issue][!shouldfail]") {
  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  highs.setOptionValue("solver", kHiPdlpString);
  highs.setOptionValue("pdlp_iteration_limit", 1000000);
  REQUIRE(highs.readModel(dataFile("weakness-cancelling-constant.mps")) ==
          HighsStatus::kOk);
  highs.run();
  // latest: Iteration limit reached after 1000000 PDLP iterations
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
}
