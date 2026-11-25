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
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <utility>
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
  }
}

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

namespace {
// splitmix64: integer only, so that the model is the same on every platform
struct SplitMix64 {
  uint64_t x;
  uint64_t next() {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  HighsInt below(HighsInt n) { return HighsInt(next() % uint64_t(n)); }
};

// Level 2 of a lexicographic pair on a multi-commodity network flow LP
// with side constraints: per commodity, flow conservation on a
// time-expanded network over num_location locations and num_period
// periods, with a supply of num_unit to 2 * num_unit - 1 units; one row
// per task (= 1) over the commodities allowed for it; minimise cost B
// subject to cost_A'x <= level1_bound. Every commodity may serve every
// task; 80% of the task arcs have cost A 1, 5% have cost B 1; all arcs have
// upper bound 79.
HighsLp lexLevel2(const HighsInt num_location, const HighsInt num_period,
                  const HighsInt num_commodity, const HighsInt num_unit,
                  const uint64_t seed, const double level1_bound) {
  const double kArcUpper = 79;
  const HighsInt kCostAPct = 80;
  const HighsInt kExtraCommodityPct = 100;
  const HighsInt kCostBPct = 5;
  SplitMix64 rng{seed};
  std::vector<HighsInt> supply;
  for (HighsInt f = 0; f < num_commodity; f++)
    supply.push_back(num_unit + rng.below(num_unit));
  const HighsInt horizon = num_period * 1440;
  struct Task {
    HighsInt orig, dest, dep, ready, commodity;
  };
  // Tasks come from generated paths of the units, so the LP is feasible
  std::vector<Task> tasks;
  for (HighsInt f = 0; f < num_commodity; f++) {
    for (HighsInt k = 0; k < supply[f]; k++) {
      HighsInt s = rng.below(num_location);
      HighsInt t = rng.below(120);
      while (true) {
        HighsInt d = rng.below(num_location - 1);
        if (d >= s) d++;
        const HighsInt dur = 45 + rng.below(240);
        if (t + dur + 30 >= horizon) break;
        tasks.push_back({s, d, t, t + dur + 30, f});
        s = d;
        t = t + dur + 30 + rng.below(180);
      }
    }
  }
  std::vector<std::vector<HighsInt>> allowed;
  for (const Task& tk : tasks) {
    std::set<HighsInt> al = {tk.commodity};
    for (HighsInt g = 0; g < num_commodity; g++)
      if (g != tk.commodity && rng.below(100) < kExtraCommodityPct)
        al.insert(g);
    allowed.emplace_back(al.begin(), al.end());
  }
  // Event times per (commodity, location)
  std::vector<std::set<HighsInt>> events(num_commodity * num_location,
                                         std::set<HighsInt>{0, horizon});
  for (size_t i = 0; i < tasks.size(); i++) {
    for (HighsInt f : allowed[i]) {
      events[f * num_location + tasks[i].orig].insert(tasks[i].dep);
      events[f * num_location + tasks[i].dest].insert(tasks[i].ready);
    }
  }
  HighsLp lp;
  // Rows: supply per commodity, node rows, task rows, the level-1 bound
  for (HighsInt f = 0; f < num_commodity; f++) {
    lp.row_lower_.push_back(supply[f]);
    lp.row_upper_.push_back(supply[f]);
  }
  std::vector<std::map<HighsInt, HighsInt>> node_row(num_commodity *
                                                     num_location);
  for (HighsInt fs = 0; fs < num_commodity * num_location; fs++) {
    for (HighsInt t : events[fs]) {
      node_row[fs][t] = lp.row_lower_.size();
      lp.row_lower_.push_back(0);
      lp.row_upper_.push_back(0);
    }
  }
  const HighsInt first_task_row = lp.row_lower_.size();
  for (size_t i = 0; i < tasks.size(); i++) {
    lp.row_lower_.push_back(1);
    lp.row_upper_.push_back(1);
  }
  const HighsInt bound_row = lp.row_lower_.size();
  lp.row_lower_.push_back(-kHighsInf);
  lp.row_upper_.push_back(level1_bound);
  lp.num_row_ = lp.row_lower_.size();

  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0};
  auto addColumn = [&](const double cost,
                       const std::vector<std::pair<HighsInt, double>>& ents) {
    lp.col_cost_.push_back(cost);
    lp.col_lower_.push_back(0);
    lp.col_upper_.push_back(kArcUpper);
    for (const auto& e : ents) {
      lp.a_matrix_.index_.push_back(e.first);
      lp.a_matrix_.value_.push_back(e.second);
    }
    lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
  };
  // Start, ground and end arcs
  for (HighsInt f = 0; f < num_commodity; f++) {
    for (HighsInt s = 0; s < num_location; s++) {
      const std::map<HighsInt, HighsInt>& rows = node_row[f * num_location + s];
      addColumn(0, {{f, 1}, {rows.at(0), 1}});
      for (auto it = rows.begin(); std::next(it) != rows.end(); ++it)
        addColumn(0, {{it->second, -1}, {std::next(it)->second, 1}});
      addColumn(0, {{rows.at(horizon), -1}});
    }
  }
  // Task arcs
  for (size_t i = 0; i < tasks.size(); i++) {
    const Task& tk = tasks[i];
    for (HighsInt f : allowed[i]) {
      const bool cost_a = rng.below(100) < kCostAPct;
      const bool cost_b = rng.below(100) < kCostBPct;
      std::vector<std::pair<HighsInt, double>> ents = {
          {node_row[f * num_location + tk.orig].at(tk.dep), -1},
          {node_row[f * num_location + tk.dest].at(tk.ready), 1},
          {first_task_row + HighsInt(i), 1}};
      if (cost_a) ents.push_back({bound_row, 1});
      addColumn(cost_b ? 1 : 0, ents);
    }
  }
  lp.num_col_ = lp.col_cost_.size();
  return lp;
}
}  // namespace

TEST_CASE("ipx-recentring-short-steps", "[highs_ipx]") {
  // Level 2 of a lexicographic pair: the level-1 objective is bounded by
  // its optimum, so that row is tight on the whole feasible set, which has
  // no interior. Near the optimum, IPX takes runs of short steps with the
  // residuals within tolerance while it recentres the iterate. It used to
  // stop after five of them with "no progress" (for 3 of these 7 bounds,
  // which are within a relative 3e-13 of the level-1 optimum), one or a few
  // steps before it took long steps again.
  const double opt1 = 325.6950297420795;
  HighsInt num_not_optimal = 0;
  std::ostringstream outcomes;
  for (HighsInt k = -3; k <= 3; k++) {
    const double bound = opt1 * (1 + k * 1e-13);
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("solver", kIpxString);
    highs.setOptionValue("run_crossover", kHighsOffString);
    REQUIRE(highs.passModel(lexLevel2(20, 3, 6, 5, 5, bound)) ==
            HighsStatus::kOk);
    REQUIRE(highs.getLp().num_row_ == 9064);
    REQUIRE(highs.getLp().num_col_ == 12612);
    highs.run();
    const HighsModelStatus status = highs.getModelStatus();
    outcomes << "\nbound " << std::setprecision(17) << bound << ": "
             << highs.modelStatusToString(status);
    if (status != HighsModelStatus::kOptimal) num_not_optimal++;
  }
  INFO(outcomes.str());
  REQUIRE(num_not_optimal == 0);

  // With the bound below the optimum by a relative 1e-7, the LP is
  // infeasible: the objective gap then grows during the run of short
  // steps, and IPX still stops with "no progress" rather than carrying on.
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kIpxString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  REQUIRE(highs.passModel(lexLevel2(20, 3, 6, 5, 5, opt1 * (1 - 1e-7))) ==
          HighsStatus::kOk);
  highs.run();
  REQUIRE(highs.getModelStatus() != HighsModelStatus::kOptimal);
  REQUIRE(highs.getInfo().ipm_iteration_count < 100);

  highs.resetThreadScheduler(true);
}
