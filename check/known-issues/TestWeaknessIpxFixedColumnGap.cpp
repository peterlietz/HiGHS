// Unreported (found when analysing #3744)
// IPX leaves the cost of fixed columns out of the objectives by which it
// normalises its gap, so it stops early when that cost cancels the rest of
// the objective, and HiGHS reports Unknown
//
// IPX removes fixed columns itself, and Iterate::optimal measures the gap
// relative to the objectives of the LP without them (it subtracts offset_,
// the cost of the fixed and implied columns). HiGHS's final check measures
// it relative to the objectives of the whole model. If the fixed columns'
// cost cancels the rest of the objective, IPX's measure is large (here
// about 1.5e5) while HiGHS's is about 1. This is what presolve does to
// HiPO in TestIssue3715.cpp, inside IPX. With HiGHS presolve on, the fixed
// column is removed before IPX, its cost goes into the offset that IPX
// includes, and IPX solves the LP.
//
// The LP: the balanced transportation LP of TestIssue3715.cpp, with its
// objective constant -147155 written as the cost of a column fixed at 1,
// so the optimum is 0 and the model has no offset. Not related to #3744:
// the final check is the same before and after it.
//
// Symptom on latest (1916a10d9d), solver=ipx, presolve off, crossover off:
// Unknown after 10 IPM iterations, P-D objective error 5.1e-5 (before
// #3744: Unknown, 1.3e-4). With presolve on: Optimal, 5.1e-9.
//
// Fix: branch fixes/ipx-fixed-column-gap (normalise IPX's gap by the
// user's objective after postprocessing, with the fixed columns).
#include <vector>

#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

static HighsLp fixedColumnCancelsLp() {
  const std::vector<double> supply = {7267, 14623, 6263};
  const std::vector<double> demand = {7615, 5870, 6327, 8341};
  HighsLp lp;
  lp.num_col_ = 13;
  lp.num_row_ = 7;
  lp.col_cost_ = {9, 4, 7, 1, 8, 8, 8, 7, 8, 4, 7, 2, -147155};
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_.assign(lp.num_col_, kHighsInf);
  lp.col_lower_[12] = 1;
  lp.col_upper_[12] = 1;
  lp.row_lower_ = supply;
  lp.row_lower_.insert(lp.row_lower_.end(), demand.begin(), demand.end());
  lp.row_upper_ = lp.row_lower_;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0};
  for (HighsInt i = 0; i < 3; ++i) {
    for (HighsInt j = 0; j < 4; ++j) {
      lp.a_matrix_.index_.push_back(i);
      lp.a_matrix_.index_.push_back(3 + j);
      lp.a_matrix_.value_.push_back(1);
      lp.a_matrix_.value_.push_back(1);
      lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
    }
  }
  // The fixed column has no entries
  lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
  return lp;
}

TEST_CASE("weakness-ipx-fixed-column-gap", "[known-issue][!shouldfail]") {
  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  highs.setOptionValue("solver", kIpxString);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  REQUIRE(highs.passModel(fixedColumnCancelsLp()) == HighsStatus::kOk);
  highs.run();
  // latest: Unknown
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  double optimality_tolerance;
  highs.getOptionValue("optimality_tolerance", optimality_tolerance);
  // latest: 5.1e-5
  REQUIRE(highs.getInfo().primal_dual_objective_error <=
          optimality_tolerance);
}
