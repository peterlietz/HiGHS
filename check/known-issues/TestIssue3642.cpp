// ERGO-Code/HiGHS#3642 (open)
// v1.16.0: wrong "Optimal" with a MIP start, presolve on (since #3523)
//
// Symptom on latest: given a feasible start of objective 24075, HiGHS
// reports "Optimal" with objective 17275. The optimum is 18150 (found
// without the start, or with presolve off).
//
// Cause: HighsMipSolverData::checkAddSolution takes the objective of the
// start's image in the reduced space as the start's value and as the upper
// bound, but keeps the start itself as the solution, and does not check that
// the image is feasible in the reduced model. A dual reduction (parallel
// columns) fixes a column that the start has at 1.
//
// Fix: branch fixes/3642-mip-start-image. The model and start are the
// issue's attachments.
#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

TEST_CASE("issue-3642-mip-start-image-infeasible-after-presolve",
          "[known-issue][!shouldfail]") {
  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  highs.setOptionValue("time_limit", 60.0);
  highs.setOptionValue("read_solution_file",
                       dataFile("issue-3642-start.sol"));
  REQUIRE(highs.readModel(dataFile("issue-3642.mps.gz")) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  // latest: 17275, below the optimum
  REQUIRE(highs.getInfo().objective_function_value == Approx(18150));
}
