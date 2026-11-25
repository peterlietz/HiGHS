#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "HCheckConfig.h"
#include "HConfig.h"
#include "Highs.h"
#include "HighsExternalApi.h"
#include "catch.hpp"
#include "io/Filereader.h"
#include "ipm/hipo/factorhighs/DenseFact.h"
#include "ipm/hipo/factorhighs/FactorHighs.h"
#include "ipm/hipo/factorhighs/FactorHighs_c_api.h"
#include "ipm/hipo/ipm/Solver.h"
#include "ipm/hipo/ipm/Status.h"
#include "lp_data/HighsCallback.h"
#include "parallel/HighsParallel.h"
#include "util/stringutil.h"

const bool dev_run = false;

void runHipoTest(
    Highs& highs, const std::string& model, const double expected_obj,
    const HighsModelStatus& expected_model_status = HighsModelStatus::kOptimal,
    const std::string& presolve = kHighsOnString) {
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("timeless_log", kHighsOnString);
  highs.setOptionValue("presolve", presolve);

  std::string filename = std::string(HIGHS_DIR) + "/check/instances/" + model;
  highs.readModel(filename);

  HighsStatus status = highs.run();
  REQUIRE(status == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == expected_model_status);

  if (expected_model_status == HighsModelStatus::kOptimal) {
    const double actual_obj = highs.getObjectiveValue();
    REQUIRE(std::abs(actual_obj - expected_obj) / std::abs(expected_obj) <
            1e-4);
  }
}

TEST_CASE("test-hipo-afiro", "[highs_hipo]") {
  Highs highs;
  runHipoTest(highs, "afiro.mps", -464.753);
  highs.resetThreadScheduler(true);
}

TEST_CASE("test-hipo-deterministic", "[highs_hipo]") {
  // Test that hipo finds the exact same solution if run twice

  std::string model = "80bau3b.mps";
  const double expected_obj = 9.8722e5;

  HighsInt iter_1, iter_2;
  HighsSolution solution_1, solution_2;

  Highs highs;
  highs.setOptionValue(kRunCrossoverString, kHighsOffString);

  runHipoTest(highs, model, expected_obj);
  solution_1 = highs.getSolution();
  iter_1 = highs.getInfo().ipm_iteration_count;

  runHipoTest(highs, model, expected_obj);
  solution_2 = highs.getSolution();
  iter_2 = highs.getInfo().ipm_iteration_count;

  REQUIRE(iter_1 == iter_2);
  REQUIRE(solution_1.value_valid == solution_2.value_valid);
  REQUIRE(solution_1.dual_valid == solution_2.dual_valid);
  REQUIRE(solution_1.col_value == solution_2.col_value);
  REQUIRE(solution_1.row_value == solution_2.row_value);
  REQUIRE(solution_1.col_dual == solution_2.col_dual);
  REQUIRE(solution_1.row_dual == solution_2.row_dual);
}

TEST_CASE("test-hipo-options", "[highs_hipo]") {
  // test all combinations of options for hipo

  std::string model = "adlittle.mps";
  const double expected_obj = 2.2549e5;
  Highs highs;

  std::vector<std::string> orders = {kHighsChooseString, kHipoMetisString,
                                     kHipoAmdString, kHipoRcmString};
  std::vector<std::string> systems = {kHighsChooseString, kHipoNormalEqString,
                                      kHipoAugmentedString};
  std::vector<std::string> parallels = {kHighsOnString, kHighsOffString,
                                        kHighsChooseString};
  std::vector<std::string> partypes = {kHipoTreeString, kHipoNodeString,
                                       kHipoBothString};

  for (auto& order : orders) {
    highs.setOptionValue(kHipoOrderingString, order);
    for (auto& system : systems) {
      highs.setOptionValue(kHipoSystemString, system);
      for (auto& parallel : parallels) {
        highs.setOptionValue(kParallelString, parallel);
        for (auto& partype : partypes) {
          highs.setOptionValue(kHipoParallelString, partype);
          runHipoTest(highs, model, expected_obj);
        }
      }
    }
  }

  highs.resetThreadScheduler(true);
}

TEST_CASE("test-hipo-qp", "[highs_hipo]") {
  Highs highs;
  runHipoTest(highs, "qptestnw.lp", -6.4500);
  runHipoTest(highs, "qjh.lp", -5.2500);
  runHipoTest(highs, "primal1.mps", -3.501296e-2);
  highs.resetThreadScheduler(true);
}

TEST_CASE("test-hipo-infeas", "[highs_hipo]") {
  const HighsModelStatus expected_status = HighsModelStatus::kInfeasible;
  Highs highs;
  runHipoTest(highs, "bgetam.mps", 0, expected_status, "off");
  runHipoTest(highs, "forest6.mps", 0, expected_status, "off");
  runHipoTest(highs, "klein1.mps", 0, expected_status, "off");
  highs.resetThreadScheduler(true);
}

TEST_CASE("test-hipo-freevar", "[highs_hipo]") {
  const HighsModelStatus expected_status = HighsModelStatus::kOptimal;
  Highs highs;
  runHipoTest(highs, "perold.mps", -9.381e3, expected_status, "on");
  runHipoTest(highs, "perold.mps", -9.381e3, expected_status, "off");
  highs.resetThreadScheduler(true);
}

TEST_CASE("issue-3369", "[highs_hipo]") {
  // min   x
  // s.t.  x >= 1
  //       . <= 5 (empty row)

  HighsLp lp;
  lp.num_col_ = 1;
  lp.num_row_ = 2;
  lp.col_cost_ = {1};
  lp.col_lower_ = {-kHighsInf};
  lp.col_upper_ = {+kHighsInf};
  lp.row_lower_ = {1, -kHighsInf};
  lp.row_upper_ = {+kHighsInf, 5};
  lp.a_matrix_.start_ = {0, 1};
  lp.a_matrix_.index_ = {0};
  lp.a_matrix_.value_ = {1};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;

  Highs h;
  h.setOptionValue("output_flag", dev_run);
  REQUIRE(h.passModel(lp) == HighsStatus::kOk);
  h.setOptionValue("solver", "hipo");
  h.setOptionValue("run_crossover", "off");
  h.setOptionValue("presolve", "off");
  HighsStatus status = h.run();
  REQUIRE(status == HighsStatus::kOk);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);

  // hipo used to fail after many iterations and restart with ipx because the
  // slack of the empty row was set incorrectly in the post-processing.
  REQUIRE(h.getInfo().ipm_iteration_count < 10);
}

TEST_CASE("issue-3749", "[highs_hipo]") {
  // max x0 + x1 - x0^2 - x1^2 - x0 x1
  // s.t. x0 + x1 <= 10, 0 <= x <= 10
  // optimum 1/3 at (1/3, 1/3)
  HighsModel model1;
  model1.lp_.num_col_ = 2;
  model1.lp_.num_row_ = 1;
  model1.lp_.col_cost_ = {1, 1};
  model1.lp_.col_lower_ = {0, 0};
  model1.lp_.col_upper_ = {10, 10};
  model1.lp_.row_lower_ = {-kHighsInf};
  model1.lp_.row_upper_ = {10};
  model1.lp_.a_matrix_.start_ = {0, 1, 2};
  model1.lp_.a_matrix_.index_ = {0, 0};
  model1.lp_.a_matrix_.value_ = {1, 1};
  model1.hessian_.dim_ = 2;
  model1.hessian_.format_ = HessianFormat::kTriangular;
  model1.hessian_.start_ = {0, 2, 3};
  model1.hessian_.index_ = {0, 1, 1};
  model1.hessian_.value_ = {-2, -1, -2};

  // max x0 + 2 x1 - 0.5 x2 + 0.5 x'Qx, Q = [-2 -1 0; -1 -3 1; 0 1 -2]
  // s.t. x0 + x1 + x2 <= 3, 0 <= x0, x1 <= 10, x2 = 2
  // optimum -2.5 at (0, 1, 2)
  HighsModel model2;
  model2.lp_.num_col_ = 3;
  model2.lp_.num_row_ = 1;
  model2.lp_.col_cost_ = {1, 2, -0.5};
  model2.lp_.col_lower_ = {0, 0, 2};
  model2.lp_.col_upper_ = {10, 10, 2};
  model2.lp_.row_lower_ = {-kHighsInf};
  model2.lp_.row_upper_ = {3};
  model2.lp_.a_matrix_.start_ = {0, 1, 2, 3};
  model2.lp_.a_matrix_.index_ = {0, 0, 0};
  model2.lp_.a_matrix_.value_ = {1, 1, 1};
  model2.hessian_.dim_ = 3;
  model2.hessian_.format_ = HessianFormat::kTriangular;
  model2.hessian_.start_ = {0, 2, 4, 5};
  model2.hessian_.index_ = {0, 1, 1, 2, 2};
  model2.hessian_.value_ = {-2, -1, -3, 1, -2};

  const std::vector<std::pair<HighsModel, double>> tests = {{model1, 1.0 / 3.0},
                                                            {model2, -2.5}};
  const double tol = 1e-6;

  auto solve = [](const HighsModel& model, HighsSolution& solution, double& obj,
                  HighsInt& iter) {
    Highs h;
    h.setOptionValue("output_flag", dev_run);
    h.setOptionValue("solver", kHipoString);
    h.setOptionValue("presolve", kHighsOffString);
    h.setOptionValue("time_limit", 5);
    REQUIRE(h.passModel(model) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
    solution = h.getSolution();
    obj = h.getObjectiveValue();
    iter = h.getInfo().ipm_iteration_count;
    h.resetThreadScheduler(true);
  };

  for (const auto& test : tests) {
    HighsModel max_model = test.first;
    max_model.lp_.sense_ = ObjSense::kMaximize;

    HighsModel min_model = test.first;
    min_model.lp_.sense_ = ObjSense::kMinimize;
    for (double& val : min_model.lp_.col_cost_) val = -val;
    for (double& val : min_model.hessian_.value_) val = -val;

    HighsSolution max_sol, min_sol;
    double max_obj, min_obj;
    HighsInt max_iter, min_iter;
    solve(max_model, max_sol, max_obj, max_iter);
    solve(min_model, min_sol, min_obj, min_iter);

    REQUIRE(std::abs(max_obj - test.second) < tol);
    REQUIRE(std::abs(max_obj + min_obj) < tol);
    REQUIRE(max_iter == min_iter);

    // primal values coincide, duals have opposite sign
    for (HighsInt i = 0; i < max_model.lp_.num_col_; ++i) {
      REQUIRE(std::abs(max_sol.col_value[i] - min_sol.col_value[i]) < tol);
      REQUIRE(std::abs(max_sol.col_dual[i] + min_sol.col_dual[i]) < tol);
    }
    for (HighsInt i = 0; i < max_model.lp_.num_row_; ++i) {
      REQUIRE(std::abs(max_sol.row_value[i] - min_sol.row_value[i]) < tol);
      REQUIRE(std::abs(max_sol.row_dual[i] + min_sol.row_dual[i]) < tol);
    }
  }
}

// A stop callback that returns nonzero stops the METIS and AMD orderings,
// which then fail. HiPO uses this for its time limit. The matrix is a grid
// Laplacian, large enough that both orderings check the callback
TEST_CASE("test-hipo-ordering-stop", "[highs_hipo]") {
  if (!HighsExternalApi::isAvailable<HighsExtras::metis>() ||
      !HighsExternalApi::isAvailable<HighsExtras::amd>())
    return;
  // Lower triangle, with the diagonal, in CSC format
  const hipo::Int m = 300;
  const hipo::Int n = m * m;
  std::vector<hipo::Int> ptr = {0};
  std::vector<hipo::Int> rows;
  for (hipo::Int k = 0; k < n; k++) {
    rows.push_back(k);
    if (k % m < m - 1) rows.push_back(k + 1);
    if (k + m < n) rows.push_back(k + m);
    ptr.push_back(static_cast<hipo::Int>(rows.size()));
  }
  const hipo::Int nz = static_cast<hipo::Int>(rows.size());
  std::vector<hipo::Int> perm(n);

  hipo::FHsolver FH;
  REQUIRE(FH.reorderMetis(n, nz, rows.data(), ptr.data(), perm.data(), false,
                          0) == 0);
  REQUIRE(FH.reorderAmd(n, nz, rows.data(), ptr.data(), perm.data(), false) ==
          0);

  FH.setStopCallback([](void*) { return 1; });
  REQUIRE(FH.reorderMetis(n, nz, rows.data(), ptr.data(), perm.data(), false,
                          0) != 0);
  REQUIRE(FH.reorderAmd(n, nz, rows.data(), ptr.data(), perm.data(), false) !=
          0);
}

// Balanced transportation LP: supply[i] from row i, demand[j] from row
// num_supply + j, cost[i * num_demand + j] for the flow from i to j
static HighsLp transportationLp(const std::vector<double>& supply,
                                const std::vector<double>& demand,
                                const std::vector<double>& cost) {
  const HighsInt num_supply = supply.size();
  const HighsInt num_demand = demand.size();
  HighsLp lp;
  lp.num_col_ = num_supply * num_demand;
  lp.num_row_ = num_supply + num_demand;
  lp.col_cost_ = cost;
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_.assign(lp.num_col_, kHighsInf);
  lp.row_lower_ = supply;
  lp.row_lower_.insert(lp.row_lower_.end(), demand.begin(), demand.end());
  lp.row_upper_ = lp.row_lower_;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0};
  for (HighsInt i = 0; i < num_supply; ++i) {
    for (HighsInt j = 0; j < num_demand; ++j) {
      lp.a_matrix_.index_.push_back(i);
      lp.a_matrix_.index_.push_back(num_supply + j);
      lp.a_matrix_.value_.push_back(1);
      lp.a_matrix_.value_.push_back(1);
      lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
    }
  }
  return lp;
}

static void solveWithHipo(Highs& h, const HighsLp& lp,
                          const std::string& presolve = kHighsOffString) {
  h.setOptionValue("output_flag", dev_run);
  REQUIRE(h.passModel(lp) == HighsStatus::kOk);
  h.setOptionValue("solver", kHipoString);
  h.setOptionValue("run_crossover", kHighsOffString);
  h.setOptionValue("presolve", presolve);
  h.run();
}

static void requireOptimalWithinTolerance(Highs& h) {
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  double optimality_tolerance;
  h.getOptionValue("optimality_tolerance", optimality_tolerance);
  REQUIRE(h.getInfo().primal_dual_objective_error <= optimality_tolerance);
}

static HighsLp issue3715Lp() {
  return transportationLp({7267, 14623, 6263}, {7615, 5870, 6327, 8341},
                          {9, 4, 7, 1, 8, 8, 8, 7, 8, 4, 7, 2});
}

TEST_CASE("hipo-issue-3715", "[highs_hipo]") {
  // The objective constant cancels c'x at the optimum, so HiPO's gap,
  // normalised without the offset, used to stop with HiGHS's P-D objective
  // error at 82.6 times optimality_tolerance
  HighsLp lp = issue3715Lp();
  lp.offset_ = -147155;
  for (const std::string presolve : {kHighsOffString, kHighsOnString}) {
    Highs h;
    solveWithHipo(h, lp, presolve);
    requireOptimalWithinTolerance(h);
    h.resetThreadScheduler(true);
  }
}

TEST_CASE("hipo-presolve-offset", "[highs_hipo]") {
  // The constant of hipo-issue-3715 written as a column fixed at 1, so that
  // presolve creates the offset: the presolved LP's offset-free objectives
  // are large while the original model's are about 0. HiPO's stop check must
  // judge the point against the original model's objectives, as HiGHS's
  // final check after postsolve does (else: Unknown, P-D objective error
  // 7.6e-4). Also as a maximisation, where the presolved LP is in the
  // opposite sense to the model. The model has an offset of its own, so that
  // a reference offset with the wrong sign is noticed
  for (const bool maximise : {false, true}) {
    HighsLp lp = issue3715Lp();
    lp.col_cost_.push_back(-147155);
    lp.col_lower_.push_back(1);
    lp.col_upper_.push_back(1);
    lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
    lp.num_col_++;
    lp.offset_ = -1e5;
    if (maximise) {
      for (double& cost : lp.col_cost_) cost = -cost;
      lp.offset_ = -lp.offset_;
      lp.sense_ = ObjSense::kMaximize;
    }
    for (const std::string presolve : {kHighsOnString, kHighsOffString}) {
      Highs h;
      solveWithHipo(h, lp, presolve);
      requireOptimalWithinTolerance(h);
      REQUIRE(std::fabs(h.getInfo().objective_function_value - lp.offset_) <
              1e-3);
      h.resetThreadScheduler(true);
    }
  }
}

TEST_CASE("hipo-large-model-offset", "[highs_hipo]") {
  // Flows of about 3e8 and an offset in the model that cancels c'x: HiPO
  // must not iterate for an accuracy it cannot reach
  //
  // splitmix64: integer only, so that the model is the same on every platform
  uint64_t state = 10;
  auto next = [&state]() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  const HighsInt num_supply = 5, num_demand = 8;
  const int64_t scale = 300000000;
  std::vector<double> supply, demand, cost;
  int64_t total = 0;
  for (HighsInt i = 0; i < num_supply; ++i) {
    const int64_t value = scale + int64_t(next() % scale);
    supply.push_back(double(value));
    total += value;
  }
  // Each of the first demands is at most total / num_demand, so the last
  // one is positive
  int64_t assigned = 0;
  const int64_t share = total / num_demand;
  for (HighsInt j = 0; j < num_demand; ++j) {
    const int64_t value = j + 1 < num_demand
                              ? share / 2 + int64_t(next() % (share / 2))
                              : total - assigned;
    demand.push_back(double(value));
    assigned += value;
  }
  for (HighsInt k = 0; k < num_supply * num_demand; ++k)
    cost.push_back(double(100 + next() % 900));
  HighsLp lp = transportationLp(supply, demand, cost);

  Highs h;
  h.setOptionValue("output_flag", dev_run);
  REQUIRE(h.passModel(lp) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  lp.offset_ = -h.getInfo().objective_function_value;

  for (const std::string presolve : {kHighsOffString, kHighsOnString}) {
    solveWithHipo(h, lp, presolve);
    requireOptimalWithinTolerance(h);
    REQUIRE(h.getInfo().ipm_iteration_count < 100);
  }
  h.resetThreadScheduler(true);
}

TEST_CASE("test-hipo-linear-solver", "[highs_hipo]") {
  // problem size
  const HighsInt n = 5;
  const HighsInt nz = 10;

  // define the lower triangle in CSC format, with 0-based indexing
  HighsInt ptr[n + 1] = {0, 3, 5, 8, 9, 10};
  HighsInt rows[nz] = {0, 2, 3, 1, 2, 2, 3, 4, 3, 4};
  double vals[nz] = {5, 3, 4, 3, 2, 9, -1, 1, 8, 1};

  // matrix is spd, so expect all pivots to be positive
  HighsInt signs[n] = {1, 1, 1, 1, 1};

  // rhs and expected solution
  double rhs[n] = {1, 2, 3, 4, 5};
  const double lhs[n] = {0.457627118644068, 1.118644067796610,
                         -0.677966101694915, 0.186440677966102,
                         5.677966101694915};

  // initialise with default number of threads
  const int num_threads = 0;
  int initialise_status = FactorHighs_initialise(num_threads);
  if (initialise_status != 0) return;

  void* S = FactorHighs_symbolic_create();
  void* FH = FactorHighs_create();

  // set options
  const int logging_on = dev_run;
  FactorHighs_setLogging(FH, logging_on);

  const int block_size = 64;
  FactorHighs_setBlockSize(FH, block_size);

  const int pivoting_off = 0;
  FactorHighs_setPivoting(FH, pivoting_off);

  FactorHighs_setRegularisation(FH, 0.0, 0.0);

  // compute ordering with metis
  HighsInt perm[n];
  HighsInt metis_status = FactorHighs_reorderMetis(FH, n, nz, rows, ptr, perm);
  REQUIRE(metis_status == 0);

  // perform analyse phase
  HighsInt analyse_status =
      FactorHighs_analyse(FH, S, n, nz, rows, ptr, signs, perm);
  REQUIRE(analyse_status == 0);

  // print extended statistics of symbolic factorisation
  int verbose = 1;
  FactorHighs_symbolic_print(FH, S, verbose);

  // factorise the matrix
  HighsInt factorise_status =
      FactorHighs_factorise(FH, S, n, nz, rows, ptr, vals);
  REQUIRE(factorise_status == 0);

  // compute the inertia of the factorisation
  HighsInt pos, neg, zero;
  double zero_tolerance = 1e-16;
  FactorHighs_inertia(FH, &pos, &neg, &zero, zero_tolerance);

  // triangular solve
  int solve_status = FactorHighs_solve(FH, rhs, 1);
  REQUIRE(solve_status == 0);

  // compute error
  double error = 0.0;
  for (int i = 0; i < n; ++i) error += fabs(rhs[i] - lhs[i]);
  REQUIRE(error < 1e-12);

  // terminate
  FactorHighs_symbolic_destroy(S);
  FactorHighs_destroy(FH);
  FactorHighs_terminate();
}

TEST_CASE("test-hipo-serial-blas", "[highs_hipo]") {
  // Test that HiPO forces OpenBLAS to run in serial, unless
  // hipo_serial_blas is false
  if (!HighsExternalApi::isAvailable<HighsExtras::blas>() ||
      !is_substring_case(HighsExtras::blas::getInfo()->provider, "openblas"))
    return;

  const int initial_threads = HighsExtras::blas::openblas_set_num_threads(0);

  Highs highs;
  bool hipo_serial_blas;
  highs.getOptionValue("hipo_serial_blas", hipo_serial_blas);
  REQUIRE(hipo_serial_blas);

  // By default, HiPO sets the number of OpenBLAS threads to 1
  const int multiple_threads = HighsExtras::blas::openblas_set_num_threads(2);
  runHipoTest(highs, "afiro.mps", -464.753);
  REQUIRE(HighsExtras::blas::openblas_set_num_threads(0) == 1);

  // OpenBLAS may be limited to a single thread, in which case the opt-out
  // cannot be observed
  if (multiple_threads > 1) {
    // With hipo_serial_blas false, HiPO leaves the number of threads
    // unchanged
    REQUIRE(HighsExtras::blas::openblas_set_num_threads(2) == multiple_threads);
    highs.setOptionValue("hipo_serial_blas", false);
    highs.clearSolver();
    runHipoTest(highs, "afiro.mps", -464.753);
    REQUIRE(HighsExtras::blas::openblas_set_num_threads(0) == multiple_threads);
  }

  HighsExtras::blas::openblas_set_num_threads(initial_threads);
  highs.resetThreadScheduler(true);
}

// Factorise the dense symmetric matrix M (n x n, column-major, only the upper
// triangle is used) with the dense kernel, with pivoting on and the static
// regularisation that HiPO uses for the normal equations, and per-pivot
// thresholds for dynamic regularisation. Return the pivots used (in the order
// of elimination) and the regularisation recorded for them.
struct DenseKernelResult {
  HighsInt info;
  std::vector<double> pivot;
  std::vector<double> totalreg;
  std::vector<HighsInt> swaps;
  std::vector<double> pivot_2x2;
};

static DenseKernelResult factoriseDenseKernel(HighsInt n, std::vector<double> M,
                                              std::vector<double> thresh,
                                              bool pivoting = true) {
  hipo::FHoptions options;
  options.reg_p = 1e-12;
  options.reg_d = 1e-10;
  options.pivoting = pivoting;
  std::vector<HighsInt> sign(n, 1);
  DenseKernelResult result;
  result.totalreg.assign(n, -1.0);
  result.swaps.assign(n, -1);
  result.pivot_2x2.assign(n, 0.0);
  hipo::DataCollector data;
  result.info = hipo::denseFactK(
      'U', n, M.data(), n, sign.data(), thresh.data(), result.totalreg.data(),
      result.swaps.data(), result.pivot_2x2.data(), data, options);
  // the kernel stores the reciprocals of the 1x1 pivots on the diagonal
  for (HighsInt j = 0; j < n; ++j) result.pivot.push_back(1.0 / M[j + n * j]);
  return result;
}

TEST_CASE("hipo-dense-kernel-static-regularisation", "[highs_hipo]") {
  if (!HighsExternalApi::isAvailable<HighsExtras::hipo>()) return;
  // Static regularisation is added once to a pivot, and totalreg records it.
  for (bool pivoting : {true, false}) {
    DenseKernelResult res = factoriseDenseKernel(1, {1.0}, {1e-24}, pivoting);
    REQUIRE(res.info == 0);
    REQUIRE(std::abs(res.totalreg[0] - 1e-12) < 1e-15);
    REQUIRE(std::abs(res.pivot[0] - (1.0 + res.totalreg[0])) < 1e-15);
  }
}

TEST_CASE("hipo-dense-kernel-dynamic-regularisation", "[highs_hipo]") {
  if (!HighsExternalApi::isAvailable<HighsExtras::hipo>()) return;
  // Normal equations [[a, a], [a, a]]: elimination cancels the second pivot to
  // rounding noise. With the relative threshold 1e-14 * a, it is replaced by
  // the threshold, and totalreg records the whole perturbation, so that the
  // factorisation is one of M + diag(totalreg).
  for (double a : {1e10, 5e3, 2e4}) {
    const double thresh = 1e-14 * a;
    DenseKernelResult res =
        factoriseDenseKernel(2, {a, 0.0, a, a}, {thresh, thresh});
    REQUIRE(res.info == 0);
    REQUIRE(res.swaps[0] == 0);
    REQUIRE(res.pivot_2x2[0] == 0.0);
    REQUIRE(res.totalreg[0] == (a + 1e-12) - a);
    REQUIRE(std::abs(res.pivot[1] - thresh) <= 1e-15 * thresh);
    // the cancelled pivot was at most a few ulps of a
    REQUIRE(std::abs(res.totalreg[1] - thresh) <= 1e-15 * a);
  }

  // A cancelled pivot of the wrong sign is regularised too, instead of being
  // used with its static regularisation (which made it exactly zero here).
  DenseKernelResult res = factoriseDenseKernel(1, {-2e-12}, {5e-11});
  REQUIRE(res.info == 0);
  REQUIRE(std::abs(res.pivot[0] - 5e-11) <= 1e-15 * 5e-11);
  REQUIRE(res.totalreg[0] == 5e-11 + 2e-12);
}

TEST_CASE("hipo-factor-dynamic-regularisation-recorded", "[highs_hipo]") {
  // Through FHsolver, on normal equations whose second pivot cancels to
  // rounding noise: getRegularisation includes the dynamic regularisation of
  // that pivot, so the factorisation is one of M + diag(reg), and a step of
  // iterative refinement against that matrix doesn't undo the regularisation.
  const HighsInt n = 2;
  const double a = 1e10;
  const double e = 2e-6;
  // M = [[a + e, a], [a, a]], lower triangle, CSC
  std::vector<HighsInt> ptr = {0, 2, 3};
  std::vector<HighsInt> rows = {0, 1, 1};
  std::vector<double> vals = {a + e, a, a};
  std::vector<HighsInt> signs = {1, 1};
  std::vector<HighsInt> perm = {0, 1};

  if (FactorHighs_initialise(0) != 0) return;
  std::vector<double> reg(n);
  std::vector<double> x = {1.0, 0.0};
  std::vector<double> dx(n);
  {
    hipo::FHsolver FH;
    hipo::Symbolic S;
    FH.setRegularisation(1e-12, 1e-10);
    FH.setDynamicRegularisation(hipo::kDynamicDiagRelSpd);
    REQUIRE(FH.analyse(S, n, ptr[n], rows.data(), ptr.data(), signs.data(),
                       perm.data()) == 0);
    REQUIRE(FH.factorise(S, n, ptr[n], rows.data(), ptr.data(), vals.data()) ==
            0);
    FH.getRegularisation(reg.data());
    REQUIRE(FH.solve(x.data()) == 0);
    // residual of b = (1, 0) against M + diag(reg), and its correction
    dx[0] = 1.0 - ((a + e + reg[0]) * x[0] + a * x[1]);
    dx[1] = 0.0 - (a * x[0] + (a + reg[1]) * x[1]);
    REQUIRE(FH.solve(dx.data()) == 0);
  }
  FactorHighs_terminate();

  // x is about (1e4, -1e4); recording only the static regularisation made
  // the correction about as large as x
  REQUIRE(std::abs(dx[0]) <= 0.05 * std::abs(x[0]));
  REQUIRE(std::abs(dx[1]) <= 0.05 * std::abs(x[1]));
  const double thresh = hipo::kDynamicDiagRelSpd * a;
  // a + e + 1e-12 rounds to a + e, and the second pivot is regularised
  REQUIRE(std::min(reg[0], reg[1]) == 0.0);
  REQUIRE(std::abs(std::max(reg[0], reg[1]) - thresh) <= 1e-15 * a);
}
