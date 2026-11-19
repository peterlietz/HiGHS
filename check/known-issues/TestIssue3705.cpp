// ERGO-Code/HiGHS#3705 (open)
// MIP: a feasible solution from kCallbackMipUserSolution that presolve cut
// off is dropped, bound included, so HiGHS can return a worse solution than
// the one offered
//
// Symptom on latest (repro 1 of the issue): the offered solution has
// objective 2.25, but at the next query the primal bound is still inf, and
// HiGHS holds the offer's infeasible postsolved image (objective -5).
//
// Cause: the offer is checked in the original space. If it is feasible
// there but its image in the reduced space is infeasible (it contradicts a
// dual reduction: x6 is parallel to x2 with a higher cost, so presolve fixes
// x6 = 0), HiGHS drops it and logs only "Solution violation". Its objective
// is not used as a bound.
//
// Fix: branch fixes/3705-callback-solution-bound.
#include <vector>

#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

namespace {
struct Offer {
  std::vector<double> solution;
  std::vector<double> primal_bound_at_query;
};
}  // namespace

TEST_CASE("issue-3705-user-solution-cut-off-by-presolve-dropped",
          "[known-issue][!shouldfail]") {
  const double inf = kHighsInf;
  // The MIP of #3643: x6 is parallel to x2 with a higher cost
  HighsLp lp;
  lp.num_col_ = 6;
  lp.num_row_ = 3;
  lp.col_cost_ = {3, 4, 0, -5, 4.5, 7.25};
  lp.col_lower_.assign(6, 0);
  lp.col_upper_.assign(6, 1);
  lp.integrality_.assign(6, HighsVarType::kInteger);
  lp.row_lower_ = {-inf, 1, -inf};
  lp.row_upper_ = {25, 1, 1};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 4, 7, 9};
  lp.a_matrix_.index_ = {1, 2, 3, 5, 0, 2, 4, 0, 3};
  lp.a_matrix_.value_ = {-5, 17, 13, -5, 1, 1, 1, 1, 1};

  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  // Heuristics off, so that the bound at query 1 is due to the offer
  highs.setOptionValue("mip_heuristic_effort", 0.0);
  highs.setOptionValue("mip_heuristic_run_feasibility_jump", false);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);

  Offer offer;
  offer.solution = {0, 0, 1, 1, 0, 1};  // feasible, objective 2.25
  REQUIRE(lp.objectiveValue(offer.solution) == Approx(2.25));
  // Offer the solution at the first query, record the primal bound at each
  highs.setCallback(
      [](int, const std::string&, const HighsCallbackOutput* data_out,
         HighsCallbackInput* data_in, void* user_data) {
        auto& offer = *static_cast<Offer*>(user_data);
        if (offer.primal_bound_at_query.empty()) {
          data_in->user_has_solution = true;
          data_in->user_solution = offer.solution;
        }
        offer.primal_bound_at_query.push_back(data_out->mip_primal_bound);
      },
      &offer);
  highs.startCallback(kCallbackMipUserSolution);
  REQUIRE(highs.run() == HighsStatus::kOk);

  // Query 1 is made before the root LP is solved
  REQUIRE(offer.primal_bound_at_query.size() > 1);
  // latest: inf
  REQUIRE(offer.primal_bound_at_query[1] <= 2.25 + 1e-9);
}
