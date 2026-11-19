// ERGO-Code/HiGHS#3715 (closed by #3744, which fixes it only for an offset
// that is part of the model)
// HiPO stops on an offset-free P-D gap, so HiGHS reports "Optimal" at up to
// 99x the tolerance, or Unknown
//
// The issue's LP is a balanced transportation LP whose objective constant
// cancels the optimal c'x (optimum 0). #3744 makes HiGHS's relative P-D
// objective error offset-free, like HiPO's gap, which fixes that LP. This
// test writes the constant as a column fixed at 1 instead (issue-3715
// -fixedcol.mps), so the original model has no offset and presolve creates
// it. HiPO's stop check (checkTerminationKkt) runs HiGHS's KKT check on the
// presolved LP, whose offset-free objectives are now large, so it accepts a
// point early; HiGHS's final check on the original LP, whose objective is
// about 0, then rejects it.
//
// Symptom on latest (1916a10d9d), presolve on, crossover off: Unknown after
// 7 HiPO iterations, P-D objective error 7.6e-4 (7,600 times the
// tolerance). Before #3744: Optimal at 70.6 times the tolerance. On 30
// time-space network LPs where presolve creates an offset of about 1.8e5,
// #3744 turns 6 Optimal solves into Unknown (and 1 Unknown into Optimal).
//
// Cause: HiGHS's relative objective error is normalised by the objectives
// without the offset, which changes with presolve: presolve moves the cost of
// fixed columns into the offset.
//
// Fix: branch fixes/3715-hipo-pd-objective-error
#include <string>

#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

#ifdef HIPO
using namespace known_issues;

// Regression test for the case fixed by #3744: the issue's LP, whose
// objective constant is part of the model
TEST_CASE("issue-3715-hipo-model-offset", "[known-issue-fixed]") {
  for (const std::string presolve : {kHighsOffString, kHighsOnString}) {
    Highs highs;
    highs.setOptionValue("output_flag", kDevRun);
    highs.setOptionValue("solver", kHipoString);
    highs.setOptionValue("run_crossover", kHighsOffString);
    highs.setOptionValue("presolve", presolve);
    REQUIRE(highs.readModel(dataFile("issue-3715.mps")) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double optimality_tolerance;
    highs.getOptionValue("optimality_tolerance", optimality_tolerance);
    // before #3744: 8.26e-6 (presolve off) and 7.06e-6 (on)
    REQUIRE(highs.getInfo().primal_dual_objective_error <=
            optimality_tolerance);
  }
}

TEST_CASE("issue-3715-hipo-presolve-offset", "[known-issue][!shouldfail]") {
  Highs highs;
  highs.setOptionValue("output_flag", kDevRun);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("presolve", kHighsOnString);
  REQUIRE(highs.readModel(dataFile("issue-3715-fixedcol.mps")) ==
          HighsStatus::kOk);
  highs.run();
  // latest: Unknown
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  double optimality_tolerance;
  highs.getOptionValue("optimality_tolerance", optimality_tolerance);
  // latest: 7.6e-4
  REQUIRE(highs.getInfo().primal_dual_objective_error <=
          optimality_tolerance);
}
#endif
