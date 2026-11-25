// Unreported (found replaying the dumps of a MIP engine)
// When IPX's main IPM ends with no progress, IPX returns without crossover,
// and HiGHS solves the LP again from scratch by simplex, even when the
// iterate is optimal up to the objective gap
//
// Near an optimal objective of zero, the relative objective gap can stay
// just above ipm_optimality_tol for good. On a 34350 x 106217 network flow
// LP (the root relaxation of a MIP after presolve), IPX reaches residuals
// of 3e-14 and 7e-12 and gap 1.8e-8 with branch fixes/ipx-fixed-column-gap,
// then ends with no progress; dual simplex from scratch then runs for more
// than an hour. latest stops that LP earlier as imprecise (its gap measure
// leaves the fixed columns' cost out), so crossover runs, but it shows the
// same weakness on the LP below.
//
// The LP: that LP reduced by fixing columns at an optimal vertex and
// deleting rows; names generic. Its optimum is the offset 4.8e-10, and its
// only cost is 465 on an empty column with bounds [0, 1]. IPX reaches
// residuals of 1e-15 and the optimal primal objective, but its dual
// objective keeps alternating between -2.8e-7 and 2.4e-7.
//
// Symptom on latest (4f2c13a083), solver=ipx, presolve off, crossover on:
// IPM no progress after 18 iterations, crossover not run, then 8 simplex
// iterations from scratch; Optimal.
//
// Fix: branch fixes/ipx-no-progress-crossover (no progress at an iterate
// within 10 times the feasibility tolerances and with a relative gap of at
// most 1e-6 is declared imprecise, so crossover starts from it).
#include <cmath>

#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

static HighsLp noProgressNearZeroLp() {
  HighsLp lp;
  lp.num_col_ = 14;
  lp.num_row_ = 11;
  lp.offset_ = 4.8423043352840978e-10;
  lp.col_cost_ = {0, 0, 0, 465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_ = {19, 19, 19, 1, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1};
  lp.row_lower_ = {-17.112870449780871, 18.550502564612334,  0,
                   0,                   0.97327352260981159, -0.68034983406164207,
                   0,                   1.0086235279325266,  -1,
                   -0.30154721648069616, 0.43763211483146358};
  lp.row_upper_ = lp.row_lower_;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 2, 3, 4, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 21};
  lp.a_matrix_.index_ = {10, 1, 10, 0, 2, 2, 3, 4, 5, 5, 7,
                         5,  6, 6,  8, 7, 8, 7, 9, 3, 4};
  lp.a_matrix_.value_ = {1, 1,  -1, -1, -1, 1,  -1, 1,  -1, 1, -1,
                         1, -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1};
  return lp;
}

TEST_CASE("weakness-ipx-no-progress-crossover",
          "[known-issue][!shouldfail]") {
  const HighsLp lp = noProgressNearZeroLp();
  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  highs.setOptionValue("solver", kIpxString);
  highs.setOptionValue("presolve", kHighsOffString);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(std::fabs(highs.getInfo().objective_function_value - lp.offset_) <
          1e-9);
  // latest: 8 (no crossover after the IPM's no progress)
  REQUIRE(highs.getInfo().simplex_iteration_count == 0);
}
