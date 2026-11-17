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
#include <string>
#include <vector>

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

// Runs IPX's main IPM on a Netlib LP from the starting point x = x0 (within
// the bounds), y = 0 and complementarity products mu, whose residuals are
// large compared with mu.
static ipx::Info ipxFromPoorStart(const std::string& model, const double mu,
                                  const double x0) {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/" +
                          model) == HighsStatus::kOk);
  const HighsLp& lp = highs.getLp();
  const Int n = lp.num_col_;
  const Int m = lp.num_row_;
  HighsSparseMatrix a = lp.a_matrix_;
  a.ensureColwise();
  std::vector<double> b(m);
  std::vector<char> type(m);
  for (Int i = 0; i < m; i++) {
    if (lp.row_lower_[i] == lp.row_upper_[i]) {
      type[i] = '=';
      b[i] = lp.row_upper_[i];
    } else if (lp.row_lower_[i] <= -kHighsInf) {
      type[i] = '<';
      b[i] = lp.row_upper_[i];
    } else {
      REQUIRE(lp.row_upper_[i] >= kHighsInf);
      type[i] = '>';
      b[i] = lp.row_lower_[i];
    }
  }
  std::vector<double> l(n), u(n);
  for (Int j = 0; j < n; j++) {
    l[j] = lp.col_lower_[j] <= -kHighsInf ? -INFINITY : lp.col_lower_[j];
    u[j] = lp.col_upper_[j] >= kHighsInf ? INFINITY : lp.col_upper_[j];
  }
  std::vector<Int> a_start(a.start_.begin(), a.start_.end());
  std::vector<Int> a_index(a.index_.begin(), a.index_.end());

  highs::parallel::initialize_scheduler();
  ipx::LpSolver lps;
  ipx::Parameters parameters;
  parameters.display = dev_run;
  parameters.highs_logging = false;
  parameters.run_crossover = 0;
  parameters.dualize = 0;  // a starting point needs the primal model
  parameters.time_limit = kHighsInf;
  lps.SetParameters(parameters);
  HighsCallback callback(nullptr);
  lps.SetCallback(&callback);
  REQUIRE(lps.LoadModel(n, 0.0, lp.col_cost_.data(), l.data(), u.data(), m,
                        a_start.data(), a_index.data(), a.value_.data(),
                        b.data(), type.data()) == 0);

  std::vector<double> x(n), xl(n), xu(n), zl(n), zu(n);
  std::vector<double> slack(m, 0.0), y(m, 0.0);
  for (Int j = 0; j < n; j++) {
    x[j] = std::min(std::max(x0, l[j]), u[j]);
    xl[j] = std::isfinite(l[j]) ? x[j] - l[j] : INFINITY;
    xu[j] = std::isfinite(u[j]) ? u[j] - x[j] : INFINITY;
    zl[j] = std::isfinite(l[j]) ? (xl[j] > 0 ? mu / xl[j] : 1.0) : 0.0;
    zu[j] = std::isfinite(u[j]) ? (xu[j] > 0 ? mu / xu[j] : 1.0) : 0.0;
  }
  std::vector<double> ax(m, 0.0);
  for (Int j = 0; j < n; j++)
    for (Int k = a_start[j]; k < a_start[j + 1]; k++)
      ax[a_index[k]] += a.value_[k] * x[j];
  for (Int i = 0; i < m; i++) {
    if (type[i] == '<') slack[i] = std::max(b[i] - ax[i], 1.0);
    if (type[i] == '>') slack[i] = std::min(b[i] - ax[i], -1.0);
  }
  REQUIRE(lps.LoadIPMStartingPoint(x.data(), xl.data(), xu.data(),
                                   slack.data(), y.data(), zl.data(),
                                   zu.data()) == 0);
  lps.Solve();
  const ipx::Info info = lps.GetInfo();
  if (dev_run)
    printf("%s: IPX status_ipm %d after %d iterations\n", model.c_str(),
           (int)info.status_ipm, (int)info.iter);

  // hack to reset scheduler
  Highs h;
  h.resetThreadScheduler(true);
  return info;
}

TEST_CASE("ipx-rebalancing-short-steps", "[highs_ipx]") {
  // The first steps are 2e-3 to 3e-2 long while they reduce pinf and dinf
  // faster than mu: the main IPM continues through them to the optimum
  const ipx::Info afiro = ipxFromPoorStart("afiro.mps", 0.1, 1.0);
  REQUIRE(afiro.status_ipm == IPX_STATUS_optimal);

  // The steps are 1e-10 long and leave pinf and dinf where they are while mu
  // grows: the main IPM stops after five of them
  const ipx::Info adlittle = ipxFromPoorStart("adlittle.mps", 1e-2, 1.0);
  REQUIRE(adlittle.status_ipm == IPX_STATUS_no_progress);
  REQUIRE(adlittle.iter == 5);
}
