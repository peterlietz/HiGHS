// Unreported (regression from #3744, found when analysing it)
// IPX, cuPDLP and HiPDLP stop on an LP whose objective constant dominates
// the objective, and HiGHS reports Unknown
//
// #3744 normalises HiGHS's relative P-D objective error by the objectives
// without the offset. IPX (Iterate::optimal), cuPDLP (dRelObjGap) and
// HiPDLP (relative_obj_gap in pdhg.cc) still normalise their own gap by the
// objectives with the offset. If the offset dominates the objective, their
// stop test is lenient by about |offset|, and HiGHS's final check rejects
// the point. Before #3744 both measures included the offset and agreed.
// HiPO is not affected the same way (its gap is offset-free); see
// TestIssue3715.cpp for its presolve case.
//
// The LP: the balanced transportation LP of TestIssue3715.cpp, with every
// cost lowered by 147155 / 28153 (optimal c'x / total supply), so that
// the optimal c'x is 0, and an objective constant of 1e6. No fixed column,
// so that IPX's fixed-variable gap (TestWeaknessIpxFixedColumnGap.cpp) is
// not involved.
//
// Symptoms on latest (1916a10d9d), P-D objective error, presolve on / off:
// - IPX, run_crossover off: Unknown 2.6e-4 / Optimal at 7.07 times the
//   tolerance (7.1e-7)
// - cuPDLP (solver=pdlp, default options): Unknown 8.3e-3 / Unknown 8.3e-3
// - HiPDLP (solver=hipdlp): Unknown 2.8e-4 / Unknown 2.8e-4
// Before #3744 (3f617408f0) all six solves are Optimal with errors of 4e-13
// to 4e-9. IPX with crossover on is Optimal (crossover cleans up); cuPDLP
// and HiPDLP fail with default options.
//
// Fix: branch fixes/dominant-offset-ipx-pdlp (stacked on
// fixes/3715-hipo-pd-objective-error): the wrappers give IPX, cuPDLP and
// HiPDLP the LP's offset less the reference offset of the incumbent model.
#include <string>
#include <vector>

#include "KnownIssues.h"
#include "catch.hpp"

using namespace known_issues;

static HighsLp dominantOffsetLp() {
  const std::vector<double> supply = {7267, 14623, 6263};
  const std::vector<double> demand = {7615, 5870, 6327, 8341};
  const std::vector<double> cost = {9, 4, 7, 1, 8, 8, 8, 7, 8, 4, 7, 2};
  const double shift = 147155.0 / 28153.0;
  HighsLp lp;
  lp.num_col_ = 12;
  lp.num_row_ = 7;
  lp.offset_ = 1e6;
  for (const double c : cost) lp.col_cost_.push_back(c - shift);
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_.assign(lp.num_col_, kHighsInf);
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
  return lp;
}

static void requireDominantOffsetSolved(const std::string& solver,
                                        const bool crossover_off) {
  for (const std::string presolve : {kHighsOnString, kHighsOffString}) {
    Highs highs;
    highs.setOptionValue("output_flag", kDevRun);
    highs.setOptionValue("solver", solver);
    highs.setOptionValue("presolve", presolve);
    if (crossover_off)
      highs.setOptionValue("run_crossover", kHighsOffString);
    REQUIRE(highs.passModel(dominantOffsetLp()) == HighsStatus::kOk);
    highs.run();
    // latest: Unknown, except IPX with presolve off
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double optimality_tolerance;
    highs.getOptionValue("optimality_tolerance", optimality_tolerance);
    // before #3744: at most 4.1e-9
    REQUIRE(highs.getInfo().primal_dual_objective_error <=
            optimality_tolerance);
  }
}

TEST_CASE("weakness-dominant-offset-ipx", "[known-issue][!shouldfail]") {
  requireDominantOffsetSolved(kIpxString, true);
}

TEST_CASE("weakness-dominant-offset-pdlp", "[known-issue][!shouldfail]") {
  requireDominantOffsetSolved(kPdlpString, false);
}

TEST_CASE("weakness-dominant-offset-hipdlp", "[known-issue][!shouldfail]") {
  requireDominantOffsetSolved(kHiPdlpString, false);
}
