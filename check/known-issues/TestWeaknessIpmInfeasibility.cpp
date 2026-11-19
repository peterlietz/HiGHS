// ERGO-Code/HiGHS#3746 (open)
// HiPO declares feasible LPs primal infeasible: its infeasibility test
// compares objective values, not certificates
//
// Solver::checkBadIter() declares primal infeasibility after iteration 5 if
// dobj > max(1e6 |pobj|, 1) (and dual infeasibility symmetrically). With a
// zero objective, pobj is 0 for every iterate, so any dual objective above 1
// passes the test. The three LPs below have c = 0 and integer data, and are
// feasible (simplex and IPX: Optimal). Presolve off; the result is the same
// with crossover on, since HiGHS takes the infeasibility as final.
//
// Symptom on latest:
// - zero-objective (7 x 11): at iteration 6 the primal iterate is feasible
//   (pinf 9e-13) and dobj = 38.4 with dinf 1e-2: "Primal infeasible".
// - free-variable (1 x 2, min 0 s.t. 3x >= 1961713, 3x >= -341609, x free):
//   dobj rises to 1e21 by iteration 6: "Primal infeasible".
// - far-start (2 x 2, needs x0 >= 4.2e6): dobj rises to 9e20 by iteration
//   5: "Primal infeasible".
//
// Cause: the objective test is not a certificate. A dual iterate proves
// primal infeasibility only if d = b'y + l'zl - u'zu > 0 is large compared
// with sum_j |x_j| |(A'y + zl - zu)_j| over the points x it rules out; for a
// (nearly) feasible x this ratio is at most about 1, as in zero-objective.
// For free variables, HiPO's dual objective also includes the terms of the
// artificial bounds +-1e4 that it gives them (PreprocessFreeVars), so the
// iterate certifies infeasibility of the bounded problem only
// (free-variable). In far-start, the iterate is a genuine certificate for
// points of its own magnitude (x0 about 6e5 in HiPO's scaled model), far
// from the feasible region.
//
// In a search over random LPs, nearly all with a zero objective and large
// right-hand sides (presolve off), latest declared 61 feasible LPs
// infeasible. Presolve on removes the small examples here, but the same
// happens there: of 150 such LPs with 40 rows, 60 columns and a feasible
// point of size 1e3, 3 were declared infeasible with HiPO's default
// settings.
//
// Fix: branch fixes/ipm-infeasibility-test (zero-objective and
// free-variable): requires the iterate to be such a certificate, without
// the artificial bounds, for points up to 10 times its own magnitude. None
// yet for far-start, which is the same as a true infeasibility seen from the
// iterate (klein1 is declared infeasible with a ratio of 12).
#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

#ifdef HIPO
using namespace known_issues;

namespace {
HighsLp zeroObjectiveLp() {
  HighsLp lp;
  lp.num_col_ = 11;
  lp.num_row_ = 7;
  lp.col_cost_.assign(lp.num_col_, 0);
  lp.col_lower_ = {0, 0, 0, 0, 0, -kHighsInf, 0, -kHighsInf, -kHighsInf, 0,
                   -kHighsInf};
  lp.col_upper_ = {kHighsInf, kHighsInf, 7372,      kHighsInf,
                   kHighsInf, kHighsInf, kHighsInf, kHighsInf,
                   kHighsInf, kHighsInf, kHighsInf};
  lp.row_lower_ = {3997, -24509, 41320, -1173, -14705, 29738, -211};
  lp.row_upper_ = {kHighsInf, -24509,    41320,    kHighsInf,
                   kHighsInf, 29738,     kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 3, 6, 8, 10, 11, 13, 15, 18, 20, 21};
  lp.a_matrix_.index_ = {3, 0, 6, 1, 5, 6, 0, 2, 1, 5, 1,
                         2, 3, 4, 6, 2, 4, 6, 4, 6, 1};
  lp.a_matrix_.value_ = {-4, -5, 6, -4, 8,  -7, 6, 4,  1, 1, 5,
                         -3, 9,  -9, -4, 5, -4, 1, 7, -5, -8};
  return lp;
}

HighsLp freeVariableLp() {
  HighsLp lp;
  lp.num_col_ = 1;
  lp.num_row_ = 2;
  lp.col_cost_ = {0};
  lp.col_lower_ = {-kHighsInf};
  lp.col_upper_ = {kHighsInf};
  lp.row_lower_ = {1961713, -341609};
  lp.row_upper_ = {kHighsInf, kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 2};
  lp.a_matrix_.index_ = {0, 1};
  lp.a_matrix_.value_ = {3, 3};
  return lp;
}

HighsLp farStartLp() {
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 2;
  lp.col_cost_ = {0, 0};
  lp.col_lower_ = {0, -kHighsInf};
  lp.col_upper_ = {kHighsInf, kHighsInf};
  lp.row_lower_ = {1780420, 3000134};
  lp.row_upper_ = {kHighsInf, kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 3};
  lp.a_matrix_.index_ = {1, 0, 1};
  lp.a_matrix_.value_ = {1, -9, 6};
  return lp;
}

HighsModelStatus solve(const HighsLp& lp, const std::string& solver) {
  Highs highs;
  LogCapture log;
  captureLog(highs, log);
  highs.setOptionValue("solver", solver);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  highs.run();
  return highs.getModelStatus();
}
}  // namespace

TEST_CASE("weakness-ipm-infeasibility-test-zero-objective",
          "[known-issue][!shouldfail]") {
  const HighsLp lp = zeroObjectiveLp();
  REQUIRE(solve(lp, kSimplexString) == HighsModelStatus::kOptimal);
  // latest: Infeasible at iteration 6
  REQUIRE(solve(lp, kHipoString) == HighsModelStatus::kOptimal);
}

TEST_CASE("weakness-ipm-infeasibility-test-free-variable",
          "[known-issue][!shouldfail]") {
  const HighsLp lp = freeVariableLp();
  REQUIRE(solve(lp, kSimplexString) == HighsModelStatus::kOptimal);
  // latest: Infeasible at iteration 6
  REQUIRE(solve(lp, kHipoString) == HighsModelStatus::kOptimal);
}

TEST_CASE("weakness-ipm-infeasibility-test-far-start",
          "[known-issue][!shouldfail]") {
  const HighsLp lp = farStartLp();
  REQUIRE(solve(lp, kSimplexString) == HighsModelStatus::kOptimal);
  // latest: Infeasible at iteration 5 (also with the fix branch)
  REQUIRE(solve(lp, kHipoString) == HighsModelStatus::kOptimal);
}
#endif
