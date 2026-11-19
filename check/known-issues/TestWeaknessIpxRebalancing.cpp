// Unreported (found in a code review of latest)
// IPX's main IPM stops with "no progress" while its short steps are
// rebalancing the iterate
//
// Seen on time-space network LPs with loose implied upper bounds: after the
// switch to basis preconditioning (with a singular crash basis), the first
// main-IPM steps were 0.001-0.05 long, the residuals fell faster than mu,
// and IPM::Driver stopped after five such steps, one to a few steps before
// the steps would have become long again: Unknown. Generated network LPs did
// not reach that state at the switch, so the second test case reads a
// time-space network LP reduced from one of those LPs. The main IPM is also
// entered directly
// when HiPO stalls and restarts IPX from its iterate: on the network of
// TestIssue3724.cpp (2048 nodes, seeds 4 and 6) IPX then takes five short
// steps after a singular crash basis and stops, Unknown; with the fix it is
// optimal after 29 and 30 iterations. (On 16_n14 itself IPX stays short
// with the fix too, until its limit of 25 bad steps.) To keep HiPO out of
// it, the first test case starts IPX with LoadIPMStartingPoint from a point
// whose residuals are large compared with mu.
//
// Symptom on latest, case weakness-ipx-rebalancing-no-progress: Netlib afiro
// from x = 1, y = 0, xl*zl = xu*zu = 0.1: the main IPM takes five steps of
// 2e-3 to 3e-2 while pinf and dinf fall slightly and mu grows, then stops
// with IPX_STATUS_no_progress at iteration 5. With the fix it continues and
// is optimal after 18 iterations.
//
// Symptom on latest, case weakness-ipx-rebalancing-network-lp:
// data/weakness-ipx-rebalancing.mps.gz is a time-space network LP (1060
// equality rows, 1120 columns with upper bounds 79 or 1, 2151 nonzeros +-1),
// the presolved form of a 41446 x 126246 LP reduced by delta debugging
// while latest kept failing and the fix kept solving it. With presolve off,
// the initial IPM runs iterations 0-5, the crash basis is singular, then
// the main IPM's steps barely move pinf (3.8e-3 to 3.3e-3 in iterations
// 6-11) and it stops with "no progress" at iteration 11: Unknown, objective
// 44.76 against the optimum 25.33. With the fix it is optimal after 28
// iterations (objective 25.333333, as dual simplex). On 20 random row and
// column permutations of the model, latest stopped each time and the fix
// solved each time. With presolve on, latest solves the model (18
// iterations), so the test turns presolve off.
//
// Cause: Driver counts every step shorter than 0.05 as bad and stops after
// five bad steps in a row. Short steps are expected while the residuals are
// large compared with mu: the iterate is outside the neighbourhood of the
// central path in which the infeasible IPM takes long steps, and each step
// that reduces the residuals faster than mu moves it back towards that
// neighbourhood.
//
// Fix: branch fixes/ipx-rebalancing-short-steps
#include <cmath>
#include <vector>

#include "KnownIssues.h"
#include "catch.hpp"
#include "ipm/ipx/ipx_status.h"
#include "ipm/ipx/lp_solver.h"
#include "lp_data/HighsCallback.h"
#include "parallel/HighsParallel.h"

using namespace known_issues;

TEST_CASE("weakness-ipx-rebalancing-no-progress",
          "[known-issue][!shouldfail]") {
  using Int = ipxint;
  Highs highs;
  highs.setOptionValue("output_flag", false);
  REQUIRE(highs.readModel(instanceFile("afiro.mps")) == HighsStatus::kOk);
  const HighsLp& lp = highs.getLp();
  const Int n = lp.num_col_;
  const Int m = lp.num_row_;
  HighsSparseMatrix a = lp.a_matrix_;
  a.ensureColwise();

  // afiro has equality and <= rows only
  std::vector<double> rhs(m);
  std::vector<char> constr_type(m);
  for (Int i = 0; i < m; i++) {
    if (lp.row_lower_[i] == lp.row_upper_[i]) {
      constr_type[i] = '=';
      rhs[i] = lp.row_upper_[i];
    } else {
      REQUIRE(lp.row_lower_[i] <= -kHighsInf);
      constr_type[i] = '<';
      rhs[i] = lp.row_upper_[i];
    }
  }
  std::vector<double> lb(n), ub(n);
  for (Int j = 0; j < n; j++) {
    lb[j] = lp.col_lower_[j] <= -kHighsInf ? -INFINITY : lp.col_lower_[j];
    ub[j] = lp.col_upper_[j] >= kHighsInf ? INFINITY : lp.col_upper_[j];
  }
  std::vector<Int> Ap(a.start_.begin(), a.start_.end());
  std::vector<Int> Ai(a.index_.begin(), a.index_.end());

  highs::parallel::initialize_scheduler();
  ipx::LpSolver lps;
  ipx::Parameters parameters;
  parameters.display = kDevRun;
  parameters.highs_logging = false;
  parameters.run_crossover = 0;
  parameters.dualize = 0;  // a starting point needs the primal model
  parameters.time_limit = kHighsInf;
  lps.SetParameters(parameters);
  HighsCallback callback(nullptr);
  lps.SetCallback(&callback);
  REQUIRE(lps.LoadModel(n, 0.0, lp.col_cost_.data(), lb.data(), ub.data(), m,
                        Ap.data(), Ai.data(), a.value_.data(), rhs.data(),
                        constr_type.data()) == 0);

  // x = 1, y = 0 and all complementarity products 0.1: relative to mu,
  // pinf and dinf are large
  const double mu = 0.1;
  std::vector<double> x(n), xl(n), xu(n), zl(n), zu(n);
  std::vector<double> slack(m, 0.0), y(m, 0.0);
  for (Int j = 0; j < n; j++) {
    x[j] = std::min(std::max(1.0, lb[j]), ub[j]);
    xl[j] = std::isfinite(lb[j]) ? x[j] - lb[j] : INFINITY;
    xu[j] = std::isfinite(ub[j]) ? ub[j] - x[j] : INFINITY;
    zl[j] = std::isfinite(lb[j]) ? mu / xl[j] : 0.0;
    zu[j] = std::isfinite(ub[j]) ? mu / xu[j] : 0.0;
  }
  std::vector<double> ax(m, 0.0);
  for (Int j = 0; j < n; j++)
    for (Int k = Ap[j]; k < Ap[j + 1]; k++) ax[Ai[k]] += a.value_[k] * x[j];
  for (Int i = 0; i < m; i++)
    if (constr_type[i] == '<') slack[i] = std::max(rhs[i] - ax[i], 1.0);
  REQUIRE(lps.LoadIPMStartingPoint(x.data(), xl.data(), xu.data(),
                                   slack.data(), y.data(), zl.data(),
                                   zu.data()) == 0);

  lps.Solve();
  const ipx::Info info = lps.GetInfo();
  if (kDevRun)
    printf("IPX status_ipm %d after %d iterations\n", (int)info.status_ipm,
           (int)info.iter);
  // Symptom: IPX_STATUS_no_progress after five short steps
  CHECK(info.status_ipm == IPX_STATUS_optimal);

  // reset the scheduler, as check/TestIpx.cpp does
  Highs h;
  h.resetThreadScheduler(true);
}

TEST_CASE("weakness-ipx-rebalancing-network-lp", "[known-issue][!shouldfail]") {
  Highs highs;
  LogCapture log;
  captureLog(highs, log);
  REQUIRE(highs.readModel(dataFile("weakness-ipx-rebalancing.mps.gz")) ==
          HighsStatus::kOk);
  highs.setOptionValue("solver", "ipx");
  highs.setOptionValue("run_crossover", "off");
  // the model is already presolved; presolving it again hides the symptom
  highs.setOptionValue("presolve", "off");
  highs.run();
  if (kDevRun)
    printf("model status %s, IPM iterations %d\n",
           highs.modelStatusToString(highs.getModelStatus()).c_str(),
           (int)highs.getInfo().ipm_iteration_count);
  // Symptom: the main IPM stops with "no progress" at iteration 11, Unknown
  CHECK(!log.contains("no progress"));
  CHECK(highs.getModelStatus() == HighsModelStatus::kOptimal);
}
