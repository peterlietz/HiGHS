// ERGO-Code/HiGHS#3642 (closed; fixed by #3762, with #3634 and #3751)
// v1.16.0: wrong "Optimal" with a MIP start, presolve on (since #3523)
//
// Symptom before #3762: given a feasible start of objective 24075, HiGHS
// reported "Optimal" with objective 17275. The optimum is 18150 (found
// without the start, or with presolve off).
//
// Cause: HighsMipSolverData::checkAddSolution took the objective of the
// start's image in the reduced space as the start's value and as the upper
// bound, but kept the start itself as the solution, and did not check that
// the image is feasible in the reduced model. A dual reduction (parallel
// columns) fixes a column that the start has at 1. #3762 bounds the reduced
// problem by the start's own objective instead, and uses the image only if
// it is feasible and postsolves to a better solution. The model and start
// are the issue's attachments; the first case is kept as a regression test.
//
// Still present (unreported): the upper limit is computed from the start's
// objective as if it were attained in the reduced space. With an integral
// objective, computeNewUpperLimit then cuts off everything that isn't at
// least one integral step better. If presolve fixes the start's costly
// columns, the start's objective need not be attained: in the second case,
// presolve fixes z = 0 (parallel to y, higher cost), the reduced objective
// is empty (integral, scale 1), the start's objective 0.3 gives the upper
// limit -1, and HiGHS reports the start as optimal although the optimum is
// 0. Symptom on latest (508bbcb23b): Optimal, objective 0.3; without the
// start, or with presolve off, Optimal 0.
//
// Fix: branch fixes/3642-mip-start-image (round the start's objective up
// to the next integral value for the upper limits, unless it is the
// objective of its image in the reduced space).
#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

TEST_CASE("issue-3642-mip-start-image-infeasible-after-presolve",
          "[known-issue-fixed]") {
  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  highs.setOptionValue("time_limit", 60.0);
  highs.setOptionValue("read_solution_file",
                       dataFile("issue-3642-start.sol"));
  REQUIRE(highs.readModel(dataFile("issue-3642.mps.gz")) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  // before #3762: 17275, below the optimum
  REQUIRE(highs.getInfo().objective_function_value == Approx(18150));
}

TEST_CASE("issue-3642-mip-start-unattained-objective",
          "[known-issue][!shouldfail]") {
  const double inf = kHighsInf;
  HighsLp lp;
  lp.num_col_ = 6;
  lp.num_row_ = 3;
  // Columns 4 (y) and 5 (z) are parallel, and z has the higher cost
  lp.col_cost_ = {0, 0, 0, 0, 0, 0.3};
  lp.col_lower_.assign(6, 0);
  lp.col_upper_ = {20, 20, 20, 20, inf, 1};
  lp.integrality_.assign(6, HighsVarType::kInteger);
  lp.row_lower_ = {98, -inf, 20};
  lp.row_upper_ = {98, 60, inf};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 6, 10, 14};
  lp.a_matrix_.index_ = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3};
  lp.a_matrix_.value_ = {3, 5, 7, 11, 2, 2, 2, 3, 4, 5, 4, 2, 3, 6};
  // A feasible start with z = 1, objective 0.3
  HighsSolution start;
  start.col_value = {2, 0, 5, 5, 0, 1};
  start.value_valid = true;
  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  REQUIRE(highs.setSolution(start) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  // latest: 0.3, the start's objective
  REQUIRE(highs.getInfo().objective_function_value == Approx(0).margin(1e-9));
}
