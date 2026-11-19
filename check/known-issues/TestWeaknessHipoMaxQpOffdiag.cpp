// ERGO-Code/HiGHS#3749 (open)
// HiPO applies the objective sense to only some uses of the Hessian, so it
// cycles or returns wrong solutions on maximisation QPs
//
// Each test solves a small QP as a maximisation and as the same QP written
// as a minimisation (cost, offset and Hessian negated), with HiPO and with
// the active-set QP solver (presolve off, so that HiPO sees the model).
//
// Symptom on latest:
// - max x0 + x1 - (x0^2 + x0 x1 + x1^2), x0 + x1 <= 10, 0 <= x <= 10:
//   HiPO reaches the time limit (5 s: 1,062,910 iterations, obj 0.28406,
//   x = (0.461, 0.077)); the min form takes 11 iterations and, like the
//   active-set solver, gives obj 1/3 at x = (1/3, 1/3).
// - a 3-column max QP with an off-diagonal Hessian coupling a fixed column
//   (x2 = 2) to x1: HiPO says Optimal with obj -5 at x = (0.5, 0, 2); the
//   optimum is -2.75 at x = (0, 1, 2).
// - the same QP with a diagonal Hessian: x is right, but the reduced cost
//   of the fixed column is 3.3 instead of -4.7.
// - the 138 Maros-Meszaros QPs negated into maximisations (cost, offset and
//   Hessian negated): 38 that HiPO solves as minimisations end Unknown (21),
//   Solve error (16) or at the 30 s time limit (HS53); HS35MOD and QFORPLAN
//   end "Optimal" with wrong objectives (HS35MOD 0.875 instead of 0.25).
//
// Cause: hipo::Model keeps c and the offset in minimisation form (negated by
// fillInIpxData for a maximisation) but Q as given, and multiplies Q by the
// sense where it uses it. KktMatrix::buildASstructure stores the
// off-diagonal entries of Q in the augmented system without the sense (the
// diagonal in buildASvalues has it), so for a maximisation the Newton
// directions belong to a different, indefinite problem; the residuals use
// the sense, so HiPO never converges. PreprocessFixedVars moves Q x of fixed
// columns into the minimisation-form cost, the offset and the reduced costs
// of the fixed columns without the sense too.
//
// Fix: branch fixes/hipo-maxqp-offdiag
#include <cmath>
#include <string>
#include <vector>

#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

#ifdef HIPO
using namespace known_issues;

namespace {
struct Result {
  HighsModelStatus status;
  double objective;  // of the minimisation form
  std::vector<double> col_value;
  std::vector<double> col_dual;  // of the minimisation form
  HighsInt ipm_iterations;
};

Result solve(const HighsModel& model, const std::string& solver) {
  Highs highs;
  LogCapture log;
  captureLog(highs, log);
  highs.setOptionValue("solver", solver);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("time_limit", 1.0);
  REQUIRE(highs.passModel(model) == HighsStatus::kOk);
  highs.run();
  const double sense = model.lp_.sense_ == ObjSense::kMaximize ? -1.0 : 1.0;
  Result result{highs.getModelStatus(),
                sense * highs.getInfo().objective_function_value,
                highs.getSolution().col_value, highs.getSolution().col_dual,
                highs.getInfo().ipm_iteration_count};
  for (double& val : result.col_dual) val *= sense;
  highs.resetGlobalScheduler(true);
  return result;
}

HighsModel negate(const HighsModel& model) {
  HighsModel max_model = model;
  max_model.lp_.sense_ = ObjSense::kMaximize;
  max_model.lp_.offset_ = -max_model.lp_.offset_;
  for (double& val : max_model.lp_.col_cost_) val = -val;
  for (double& val : max_model.hessian_.value_) val = -val;
  return max_model;
}

// min -x0 - 2 x1 + 0.5 x2 + 0.25 + 0.5 x'Qx, x0 + x1 + x2 <= 3,
// 0 <= x0, x1 <= 10, x2 = 2; Q couples x0-x1 and x1-x2 if offdiag
HighsModel threeColumns(const bool offdiag) {
  HighsModel model;
  HighsLp& lp = model.lp_;
  lp.num_col_ = 3;
  lp.num_row_ = 1;
  lp.col_cost_ = {-1, -2, 0.5};
  lp.offset_ = 0.25;
  lp.col_lower_ = {0, 0, 2};
  lp.col_upper_ = {10, 10, 2};
  lp.row_lower_ = {-kHighsInf};
  lp.row_upper_ = {3};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2, 3};
  lp.a_matrix_.index_ = {0, 0, 0};
  lp.a_matrix_.value_ = {1, 1, 1};
  HighsHessian& hessian = model.hessian_;
  hessian.dim_ = 3;
  hessian.format_ = HessianFormat::kTriangular;
  if (offdiag) {
    hessian.start_ = {0, 2, 4, 5};
    hessian.index_ = {0, 1, 1, 2, 2};
    hessian.value_ = {2, 1, 3, -1, 2};
  } else {
    hessian.start_ = {0, 1, 2, 3};
    hessian.index_ = {0, 1, 2};
    hessian.value_ = {2, 3, 2};
  }
  return model;
}
}  // namespace

TEST_CASE("weakness-hipo-maxqp-offdiag-cycles", "[known-issue][!shouldfail]") {
  // min -x0 - x1 + x0^2 + x0 x1 + x1^2, x0 + x1 <= 10, 0 <= x <= 10
  HighsModel min_model;
  HighsLp& lp = min_model.lp_;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {-1, -1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {10, 10};
  lp.row_lower_ = {-kHighsInf};
  lp.row_upper_ = {10};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  HighsHessian& hessian = min_model.hessian_;
  hessian.dim_ = 2;
  hessian.format_ = HessianFormat::kTriangular;
  hessian.start_ = {0, 2, 3};
  hessian.index_ = {0, 1, 1};
  hessian.value_ = {2, 1, 2};
  const HighsModel max_model = negate(min_model);

  const Result hipo_min = solve(min_model, kHipoString);
  const Result qpasm_max = solve(max_model, kQpAsmString);
  REQUIRE(hipo_min.status == HighsModelStatus::kOptimal);
  REQUIRE(hipo_min.ipm_iterations > 0);  // HiPO ran
  REQUIRE(std::abs(hipo_min.objective + 1.0 / 3) < 1e-6);
  REQUIRE(qpasm_max.status == HighsModelStatus::kOptimal);
  REQUIRE(std::abs(qpasm_max.objective + 1.0 / 3) < 1e-6);

  // latest: time limit (1 s) after 10^4-10^5 iterations, obj ~0.26
  const Result hipo_max = solve(max_model, kHipoString);
  CHECK(hipo_max.status == HighsModelStatus::kOptimal);
  CHECK(hipo_max.ipm_iterations < 50);
  REQUIRE(std::abs(hipo_max.objective + 1.0 / 3) < 1e-6);
}

TEST_CASE("weakness-hipo-maxqp-offdiag-fixed-column",
          "[known-issue][!shouldfail]") {
  const HighsModel min_model = threeColumns(true);
  const HighsModel max_model = negate(min_model);

  const Result hipo_min = solve(min_model, kHipoString);
  const Result qpasm_max = solve(max_model, kQpAsmString);
  REQUIRE(hipo_min.status == HighsModelStatus::kOptimal);
  REQUIRE(hipo_min.ipm_iterations > 0);  // HiPO ran
  REQUIRE(std::abs(hipo_min.objective - 2.75) < 1e-6);
  REQUIRE(qpasm_max.status == HighsModelStatus::kOptimal);
  REQUIRE(std::abs(qpasm_max.objective - 2.75) < 1e-6);

  // latest: "Optimal", objective 5 (max form -5) at x = (0.5, 0, 2)
  const Result hipo_max = solve(max_model, kHipoString);
  REQUIRE(hipo_max.status == HighsModelStatus::kOptimal);
  CHECK(std::abs(hipo_max.col_value[0] - 0) < 1e-6);
  CHECK(std::abs(hipo_max.col_value[1] - 1) < 1e-6);
  REQUIRE(std::abs(hipo_max.objective - 2.75) < 1e-6);
}

TEST_CASE("weakness-hipo-maxqp-fixed-column-dual",
          "[known-issue][!shouldfail]") {
  const HighsModel min_model = threeColumns(false);
  const HighsModel max_model = negate(min_model);

  const Result hipo_min = solve(min_model, kHipoString);
  const Result qpasm_max = solve(max_model, kQpAsmString);
  REQUIRE(hipo_min.status == HighsModelStatus::kOptimal);
  REQUIRE(hipo_min.ipm_iterations > 0);  // HiPO ran
  REQUIRE(std::abs(hipo_min.col_dual[2] - 4.7) < 1e-6);
  REQUIRE(qpasm_max.status == HighsModelStatus::kOptimal);
  REQUIRE(std::abs(qpasm_max.col_dual[2] - 4.7) < 1e-6);

  // latest: x and the objective are right, but the reduced cost of the
  // fixed column is 3.3 (max form) instead of -4.7
  const Result hipo_max = solve(max_model, kHipoString);
  REQUIRE(hipo_max.status == HighsModelStatus::kOptimal);
  REQUIRE(std::abs(hipo_max.objective - hipo_min.objective) < 1e-6);
  REQUIRE(std::abs(hipo_max.col_dual[2] - 4.7) < 1e-6);
}
#endif
