// ERGO-Code/HiGHS#3699 (open)
// MIP: cuts are discarded when an integral coefficient is on a column
// without a lower bound, so a 3-row infeasible MIP never terminates
//
// Symptom on latest: the search runs until the time limit (millions of
// nodes, dual bound stuck at 0). The MIP is infeasible: r1 forces y to be
// odd, r2 forces 3y to be even. SCIP proves it at node 0.
//
// Cause: HighsCutGeneration::postprocessCut compensates each coefficient's
// rounding residual delta with a column bound. An exactly integral
// coefficient (delta == 0) takes the lower-bound branch, and a column
// without a lower bound rejects the whole cut.
//
// Fix: branch fixes/3699-cut-integral-coefficient (the diff proposed in
// the issue). Upstream agreed and said it would go into the refactor of
// #3473.
#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

TEST_CASE("issue-3699-integral-cut-coefficient-on-free-column",
          "[known-issue][!shouldfail]") {
  const double inf = kHighsInf;
  // Columns x, y, u, v, all integer, objective 0
  HighsLp lp;
  lp.num_col_ = 4;
  lp.num_row_ = 3;
  lp.col_cost_ = {0, 0, 0, 0};
  lp.col_lower_ = {1, 1, -inf, -inf};
  lp.col_upper_ = {6, 3, -1999, -1998};
  lp.integrality_.assign(4, HighsVarType::kInteger);
  // r0: -3u + v >= 4000; r1: 3y - 2x = 1; r2: -2v + 2u + 3y = 0
  lp.row_lower_ = {4000, 1, 0};
  lp.row_upper_ = {inf, 1, 0};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 2, 4, 7};
  lp.a_matrix_.index_ = {2, 3, 0, 1, 1, 2, 3};
  lp.a_matrix_.value_ = {-3, 1, -2, 3, 3, 2, -2};

  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  // With the fix, infeasibility is proved at the root in about 0.01 s
  highs.setOptionValue("time_limit", 5.0);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  highs.run();
  // latest: time limit reached
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kInfeasible);
}
