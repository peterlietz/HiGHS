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

// Random min-cost flow on num_node nodes: a cycle of uncapacitated arcs (so
// that it is feasible) and degree * num_node random capacitated arcs. The
// Cholesky factor of A * D * A^T of such a graph is nearly dense.
static HighsLp randomNetworkLp(const HighsInt num_node, const HighsInt degree,
                               uint64_t seed) {
  // splitmix64, so that the model is the same on every platform
  auto next = [&seed]() {
    uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  const HighsInt m = num_node;
  HighsLp lp;
  lp.num_row_ = m;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0};
  auto addArc = [&lp](HighsInt tail, HighsInt head, double cost, double upper) {
    lp.col_cost_.push_back(cost);
    lp.col_lower_.push_back(0);
    lp.col_upper_.push_back(upper);
    lp.a_matrix_.index_.push_back(tail);
    lp.a_matrix_.value_.push_back(-1);
    lp.a_matrix_.index_.push_back(head);
    lp.a_matrix_.value_.push_back(1);
    lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
  };
  for (HighsInt i = 0; i < m; i++) addArc(i, (i + 1) % m, 1000, kHighsInf);
  for (HighsInt k = 0; k < degree * m; k++) {
    const HighsInt i = next() % m;
    HighsInt j = next() % m;
    if (i == j) j = (j + 1) % m;
    const double cost = double(1 + next() % 100);
    const double upper = double(1 + next() % 50);
    addArc(i, j, cost, upper);
  }
  lp.num_col_ = lp.col_cost_.size();
  std::vector<double> supply(m, 0);
  for (HighsInt k = 0; k < m / 10; k++) {
    const HighsInt i = next() % m;
    const HighsInt j = next() % m;
    const double v = double(1 + next() % 100);
    supply[i] += v;
    supply[j] -= v;
  }
  for (HighsInt i = 0; i < m; i++) lp.row_lower_.push_back(-supply[i]);
  lp.row_upper_ = lp.row_lower_;
  return lp;
}

TEST_CASE("test-hipo-network-ipx", "[highs_hipo]") {
  // With solver = "ipm", a network LP whose factorisation is predicted to be
  // nearly dense (1.3e10 flops, 1.8e5 per nonzero of A) is left to IPX; with
  // solver = "hipo", HiPO solves it
  std::string log_text;
  Highs highs;
  highs.setOptionValue("output_flag", true);
  highs.setOptionValue("log_to_console", dev_run);
  highs.setCallback(
      [](int callback_type, const std::string& message,
         const HighsCallbackOutput*, HighsCallbackInput*, void* user_data) {
        if (callback_type == kCallbackLogging)
          *static_cast<std::string*>(user_data) += message;
      },
      &log_text);
  highs.startCallback(kCallbackLogging);
  highs.setOptionValue("run_crossover", kHighsOffString);
  REQUIRE(highs.passModel(randomNetworkLp(6000, 5, 1)) == HighsStatus::kOk);
  const std::string switch_message = "solving it with IPX instead";

  highs.setOptionValue("solver", kIpmString);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(log_text.find(switch_message) != std::string::npos);
  REQUIRE(log_text.find("IPX model has") != std::string::npos);
  const HighsInt ipm_iteration_count = highs.getInfo().ipm_iteration_count;
  const double objective = highs.getInfo().objective_function_value;

  highs.setOptionValue("solver", kIpxString);
  highs.clearSolver();
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getInfo().ipm_iteration_count == ipm_iteration_count);
  REQUIRE(std::abs(highs.getInfo().objective_function_value - objective) <=
          1e-9 * std::abs(objective));

  log_text.clear();
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("ipm_iteration_limit", 1);
  highs.clearSolver();
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kIterationLimit);
  REQUIRE(log_text.find(switch_message) == std::string::npos);
  REQUIRE(log_text.find("IPX model has") == std::string::npos);

  // Matrices with at most two nonzeros per column that aren't network
  // matrices are left to IPX as well: their bases are pseudoforests, so IPX's
  // LU factors stay sparse. Here, a 2-matching (every column (+1, +1))
  highs.setOptionValue("solver", kIpmString);
  HighsLp lp = randomNetworkLp(6000, 5, 1);
  for (HighsInt col = 0; col < lp.num_col_; col++)
    lp.a_matrix_.value_[lp.a_matrix_.start_[col]] = 1;
  lp.row_lower_.assign(lp.num_row_, 2);
  lp.row_upper_ = lp.row_lower_;
  log_text.clear();
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kIterationLimit);
  REQUIRE(log_text.find(switch_message) != std::string::npos);
  REQUIRE(log_text.find("IPX model has") != std::string::npos);
  highs.resetThreadScheduler(true);
}
