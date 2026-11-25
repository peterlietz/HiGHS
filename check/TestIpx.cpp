#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "ipm/ipx/ipx_status.h"
#include "ipm/ipx/lp_solver.h"
#include "lp_data/HConst.h"
#include "lp_data/HighsCallback.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsStatus.h"
#include "parallel/HighsParallel.h"

// Example for using IPX from its C++ interface. The program solves the Netlib
// problem afiro.

#include <cmath>
#include <iostream>

#include "ipm/ipx/lp_solver.h"

const bool dev_run = false;

using Int = ipxint;

constexpr HighsInt num_var = 12;
constexpr HighsInt num_constr = 9;
const double offset = 0;
const double obj[] = {-0.2194, 0.0, 0.0,   0.0,     0.0, 0.0,
                      0.0,     0.0, -0.32, -0.5564, 0.6, -0.48};
const double lb[num_var] = {0.0};
const double ub[] = {80.0,     283.303,  283.303, 312.813, 349.187, INFINITY,
                     INFINITY, INFINITY, 57.201,  500.0,   500.501, 357.501};
// Constraint matrix in CSC format with 0-based indexing.
const HighsInt Ap[] = {0, 2, 6, 10, 14, 18, 20, 22, 24, 26, 28, 30, 32};
const HighsInt Ai[] = {0, 5, 1, 6, 7, 8, 2, 6, 7, 8, 3, 6, 7, 8, 4, 6,
                       7, 8, 1, 2, 2, 3, 2, 4, 0, 6, 0, 5, 2, 5, 5, 7};
const double Ax[] = {-1.0,      0.301,   1.0,   -1.0, 0.301, 1.06,    1.0,
                     -1.0,      0.313,   1.06,  1.0,  -1.0,  0.313,   0.96,
                     1.0,       -1.0,    0.326, 0.86, -1.0,  0.99078, 1.00922,
                     -1.0,      1.01802, -1.0,  1.4,  1.0,   0.109,   -1.0,
                     -0.419111, 1.0,     1.4,   -1.0};
const double rhs[] = {0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 44.0, 300.0};
const char constr_type[] = {'<', '<', '=', '<', '<', '=', '<', '<', '<'};

TEST_CASE("test-ipx", "[highs_ipx]") {
  ipx::LpSolver lps;
  ipx::Parameters parameters;
  if (!dev_run) parameters.display = 0;
  parameters.highs_logging = false;
  parameters.timeless_log = false;
  parameters.time_limit = kHighsInf;
  lps.SetParameters(parameters);

  // Solve the LP.
  Int load_status = lps.LoadModel(num_var, offset, obj, lb, ub, num_constr, Ap,
                                  Ai, Ax, rhs, constr_type);
  REQUIRE(load_status == 0);

  highs::parallel::initialize_scheduler();

  // set empty callback to avoid issues
  HighsCallback callback(nullptr);
  lps.SetCallback(&callback);

  Int status = lps.Solve();
  bool is_solved = status == IPX_STATUS_solved;
  REQUIRE(is_solved);

  // Get solver and solution information.
  ipx::Info info = lps.GetInfo();

  // Get the interior solution (available if IPM was started).
  double x[num_var], xl[num_var], xu[num_var], slack[num_constr];
  double y[num_constr], zl[num_var], zu[num_var];
  lps.GetInteriorSolution(x, xl, xu, slack, y, zl, zu);

  REQUIRE(fabs(x[11] - 339.9) < 1.0);
  REQUIRE(fabs(xl[11] - 339.94) < 1.0);
  REQUIRE(fabs(xu[11] - 17.55) < 1.0);
  REQUIRE(fabs(slack[8] - 234.76) < 1.0);
  REQUIRE(fabs(y[8]) < 1.0);
  REQUIRE(fabs(zl[11]) < 1.0);
  REQUIRE(fabs(zu[11]) < 1.0);

  // Get the basic solution (available if crossover terminated without error).
  double ipx_col_value[num_var], ipx_row_value[num_constr];
  double ipx_row_dual[num_constr], ipx_col_dual[num_var];
  Int ipx_row_status[num_constr], ipx_col_status[num_var];
  lps.GetBasicSolution(ipx_col_value, ipx_row_value, ipx_row_dual, ipx_col_dual,
                       ipx_row_status, ipx_col_status);
  REQUIRE(fabs(ipx_col_value[11] - 339.9) < 1);

  (void)(info);  // surpress unused variable.

  // hack to reset scheduler
  Highs h;
  h.resetThreadScheduler(true);
}

TEST_CASE("ipx-no-progress-crossover", "[highs_ipx]") {
  // A network flow LP whose optimum is the offset 4.8e-10, reduced from the
  // root relaxation of a MIP. Its only cost is 465 on an empty column with
  // bounds [0, 1]. IPX reaches an iterate with residuals of 1e-15 and an
  // optimal primal objective, but its dual objective keeps alternating
  // between -2.8e-7 and 2.4e-7, so the gap stays above ipm_optimality_tol
  // and the IPM ends with no progress. IPX used to return without
  // crossover then, and HiGHS solved the LP again from scratch by simplex
  // (8 iterations here; more than an hour on the unreduced LP). Now the
  // iterate is declared imprecise, and crossover starts from it.
  HighsLp lp;
  lp.num_col_ = 14;
  lp.num_row_ = 11;
  lp.offset_ = 4.8423043352840978e-10;
  lp.col_cost_ = {0, 0, 0, 465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_ = {19, 19, 19, 1, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1};
  lp.row_lower_ = {-17.112870449780871, 18.550502564612334, 0,
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
  Highs h;
  h.setOptionValue("output_flag", dev_run);
  h.setOptionValue("solver", kIpxString);
  h.setOptionValue("presolve", kHighsOffString);
  REQUIRE(h.passModel(lp) == HighsStatus::kOk);
  h.run();
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  // Without crossover from the IPX iterate: 8 simplex iterations
  REQUIRE(h.getInfo().simplex_iteration_count == 0);
  REQUIRE(std::fabs(h.getInfo().objective_function_value - lp.offset_) < 1e-9);
  h.resetThreadScheduler(true);
}

TEST_CASE("ipx-fixed-column-gap", "[highs_ipx]") {
  // A balanced transportation LP whose objective constant -147155 (optimal
  // c'x) is the cost of a column fixed at 1, so that the optimum is 0.
  // - Presolve off: IPX removes the fixed column itself and used to
  //   normalise its gap by the objective of the LP without it (about 1.5e5),
  //   so it stopped with HiGHS's P-D objective error at 5.1e-5: Unknown
  // - Presolve on, IPX dualizing the presolved LP: its objective without the
  //   offset is that of the LP negated, so the gap was normalised by about
  //   twice the offset, and IPX stopped at 14 times the tolerance
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
  lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());

  struct Run {
    std::string presolve;
    HighsInt dualize;
  };
  for (const Run& run : {Run{kHighsOffString, kIpxDualizeStrategyOff},
                         Run{kHighsOnString, kIpxDualizeStrategyOn}}) {
    const HighsInt dualize = run.dualize;
    Highs h;
    h.setOptionValue("output_flag", dev_run);
    h.setOptionValue("solver", kIpxString);
    h.setOptionValue("presolve", run.presolve);
    h.setOptionValue("run_crossover", kHighsOffString);
    h.setOptionValue("ipx_dualize_strategy", dualize);
    REQUIRE(h.passModel(lp) == HighsStatus::kOk);
    h.run();
    REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
    double optimality_tolerance;
    h.getOptionValue("optimality_tolerance", optimality_tolerance);
    REQUIRE(h.getInfo().primal_dual_objective_error <= optimality_tolerance);
    REQUIRE(std::fabs(h.getInfo().objective_function_value) < 1e-3);
    h.resetThreadScheduler(true);
TEST_CASE("ipx-stall-cancelling-constant", "[highs_ipx]") {
  // A balanced transportation LP (5 supplies of about 4e7, 8 demands) whose
  // objective constant -59861304428 cancels the optimal c'x, so that the
  // optimum is 0. After HiGHS presolve, IPX reaches an objective gap of
  // 2.2e-6, the rounding error of objectives whose terms are about 6e10, at
  // iteration 11 and keeps it, with complementarity about 3e-22 and steps
  // of 1.00 (primal) and 0.31 (dual), which its other bad-iteration rules
  // don't count. It used to iterate until the iteration or time limit. Now
  // it stops with "no progress", and simplex cleans up or HiGHS's final
  // check accepts the point
  const std::string model_file =
      std::string(HIGHS_DIR) + "/check/instances/cancelling-constant.mps";
  for (const std::string crossover : {kHighsOnString, kHighsOffString}) {
    Highs h;
    h.setOptionValue("output_flag", dev_run);
    h.setOptionValue("solver", kIpxString);
    h.setOptionValue("run_crossover", crossover);
    h.setOptionValue("ipm_iteration_limit", 1000);
    REQUIRE(h.readModel(model_file) == HighsStatus::kOk);
    h.run();
    REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(h.getInfo().ipm_iteration_count < 100);
  }
}
