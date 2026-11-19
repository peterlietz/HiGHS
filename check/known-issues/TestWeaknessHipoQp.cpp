// ERGO-Code/HiGHS#3747 (open)
// HiPO's termination is final for QP: there is no KKT check, so HiPO returns
// a point that the QP check of HiGHS then rejects
//
// The QP is min 1e6 y + y^2/2 + x^2/2 - 0.4 x  s.t.  y >= 1e5, 0 <= x <= 1,
// with optimum y = 1e5, x = 0.4 and objective 1.05e11 - 0.08.
//
// Symptom on latest: HiPO stops after 5 iterations with relative gap 7.5e-9
// and x = 0.626, where the dual of x is 0.226: "QP solver claims optimality,
// but with ... dual(1/0.2261/0.2261) infeasibilities", model status Solve
// error. --solver ipm takes the same path. The active-set solver (qpasm, the
// default for QP) returns Optimal with objective 1.05e11. In the
// Maros-Meszaros set, QGFRDXPN fails the same way (dual infeasibility 0.275
// on a [0,1] column at 0.424; objective 1.0079e11); qpasm solves it.
//
// Cause: HiPO's gap test is relative to the objective, so with a large
// objective it allows complementarity products of order 0.1, i.e. a large
// dual on a variable away from its bound. For LP, checkTermination then runs
// checkTerminationKkt (the KKT check of HiGHS) and keeps iterating if it
// fails, and IPX or crossover can still rescue a stalled solve. For QP,
// checkTerminationKkt returns "kktCheck skipped for QP", checkTermination
// accepts HiPO's own test, and refinement and crossover are off, so nothing
// catches it before solveQp's final check.
//
// Fix: branch fixes/hipo-qp (QP KKT check, the one solveQp applies, before
// HiPO accepts its own termination; 9 iterations here)
#include <cmath>

#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

#ifdef HIPO
using namespace known_issues;

namespace {
HighsModel largeObjectiveQp() {
  HighsModel model;
  HighsLp& lp = model.lp_;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {1e6, -0.4};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {kHighsInf, 1};
  lp.row_lower_ = {1e5};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 1};
  lp.a_matrix_.index_ = {0};
  lp.a_matrix_.value_ = {1};
  HighsHessian& hessian = model.hessian_;
  hessian.dim_ = 2;
  hessian.format_ = HessianFormat::kTriangular;
  hessian.start_ = {0, 1, 2};
  hessian.index_ = {0, 1};
  hessian.value_ = {1, 1};
  return model;
}
}  // namespace

TEST_CASE("weakness-hipo-qp-no-kkt-check", "[known-issue][!shouldfail]") {
  const HighsModel model = largeObjectiveQp();

  // Reference: the active-set QP solver
  Highs asm_highs;
  asm_highs.setOptionValue("output_flag", kDevRun);
  asm_highs.setOptionValue("solver", kQpAsmString);
  REQUIRE(asm_highs.passModel(model) == HighsStatus::kOk);
  REQUIRE(asm_highs.run() == HighsStatus::kOk);
  REQUIRE(asm_highs.getModelStatus() == HighsModelStatus::kOptimal);
  const double asm_obj = asm_highs.getObjectiveValue();

  Highs highs;
  LogCapture log;
  captureLog(highs, log);
  highs.setOptionValue("solver", kHipoString);
  REQUIRE(highs.passModel(model) == HighsStatus::kOk);
  // latest: returns an error, with model status Solve error
  highs.run();
  // latest: "QP solver claims optimality, but with ... dual(1/0.2261/0.2261)"
  REQUIRE_FALSE(log.contains("claims optimality"));
  // latest: Solve error
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(std::abs(highs.getObjectiveValue() - asm_obj) < 1e-9 * asm_obj);
  highs.resetThreadScheduler(true);
}
#endif
