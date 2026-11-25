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

TEST_CASE("test-hipo-qp-kkt", "[highs_hipo]") {
  // min 1e6 y + y^2/2 + x^2/2 - 0.4 x  s.t.  y >= 1e5, 0 <= x <= 1
  //
  // The objective, 1.05e11, lets HiPO's relative gap test pass while x is
  // still far from its bounds with a dual of 0.2, which the QP KKT check of
  // HiGHS reports as a dual infeasibility. HiPO has to check that itself and
  // continue, rather than return a solution that HiGHS rejects.
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

  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  REQUIRE(highs.passModel(model) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(highs.getInfo().num_dual_infeasibilities == 0);
  const double expected_obj = 1e11 + 0.5e10 - 0.08;
  REQUIRE(std::abs(highs.getObjectiveValue() - expected_obj) <
          1e-9 * expected_obj);
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

TEST_CASE("test-hipo-feasible-zero-objective", "[highs_hipo]") {
  // Feasible LPs with c = 0 that HiPO used to declare primal infeasible,
  // because a dual objective above 1 passed its objective-based test
  auto solve = [](const HighsLp& lp) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("solver", kHipoString);
    highs.setOptionValue("presolve", kHighsOffString);
    highs.setOptionValue("run_crossover", kHighsOffString);
    REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
    highs.run();
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    highs.resetThreadScheduler(true);
  };

  // The primal iterate is feasible when the dual objective passes 1
  HighsLp lp;
  lp.num_col_ = 11;
  lp.num_row_ = 7;
  lp.col_cost_.assign(lp.num_col_, 0);
  lp.col_lower_ = {0, 0,          0,          0, 0,         -kHighsInf,
                   0, -kHighsInf, -kHighsInf, 0, -kHighsInf};
  lp.col_upper_.assign(lp.num_col_, kHighsInf);
  lp.col_upper_[2] = 7372;
  lp.row_lower_ = {3997, -24509, 41320, -1173, -14705, 29738, -211};
  lp.row_upper_ = {kHighsInf, -24509, 41320,    kHighsInf,
                   kHighsInf, 29738,  kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 3, 6, 8, 10, 11, 13, 15, 18, 20, 21};
  lp.a_matrix_.index_ = {3, 0, 6, 1, 5, 6, 0, 2, 1, 5, 1,
                         2, 3, 4, 6, 2, 4, 6, 4, 6, 1};
  lp.a_matrix_.value_ = {-4, -5, 6,  -4, 8, -7, 6, 4, 1,  1, 5,
                         -3, 9,  -9, -4, 5, -4, 1, 7, -5, -8};
  solve(lp);

  // min 0 s.t. 3x >= 1961713, 3x >= -341609, x free: infeasible only within
  // HiPO's artificial bounds on the free variable
  lp = HighsLp();
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
  solve(lp);

  // Feasible QPs with c = 0 (random rows, large right-hand sides, some free
  // columns, and a column 0 <= w <= 1 with Q_ww = 1 and no rows) that HiPO
  // declared primal infeasible: the certificate test applies to QPs too
  auto solveQp = [](HighsLp lp) {
    const HighsInt w = lp.num_col_++;
    lp.col_cost_.assign(lp.num_col_, 0);
    lp.col_lower_.push_back(0);
    lp.col_upper_.push_back(1);
    lp.row_upper_.assign(lp.num_row_, kHighsInf);
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.start_.push_back(lp.a_matrix_.start_.back());
    HighsModel model;
    model.lp_ = lp;
    model.hessian_.dim_ = lp.num_col_;
    model.hessian_.format_ = HessianFormat::kTriangular;
    model.hessian_.start_.assign(lp.num_col_ + 1, 0);
    model.hessian_.start_[lp.num_col_] = 1;
    model.hessian_.index_ = {w};
    model.hessian_.value_ = {1};
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("solver", kHipoString);
    highs.setOptionValue("presolve", kHighsOffString);
    highs.setOptionValue("run_crossover", kHighsOffString);
    REQUIRE(highs.passModel(model) == HighsStatus::kOk);
    highs.run();
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    highs.resetThreadScheduler(true);
  };
  const double inf = kHighsInf;
  lp = HighsLp();
  lp.num_col_ = 12;
  lp.num_row_ = 8;
  lp.row_lower_ = {3696052, -1316217, -8115407, 3747910,
                   1339586, 4481191,  4008801,  627961};
  lp.col_lower_ = {0, 0, 0, 0, 0, 0, 0, -inf, -inf, 0, 0, 0};
  lp.col_upper_ = {inf, inf, inf, 438747, inf, inf,
                   inf, inf, inf, inf,    inf, inf};
  lp.a_matrix_.start_ = {0, 2, 3, 6, 7, 7, 13, 15, 17, 19, 22, 24, 26};
  lp.a_matrix_.index_ = {1, 2, 0, 2, 5, 6, 3, 0, 1, 2, 3, 5, 6,
                         3, 4, 2, 7, 5, 6, 3, 4, 7, 2, 3, 2, 3};
  lp.a_matrix_.value_ = {7, -5, 1, -3, 1,  6, 1,  4, -7, 1,  9,  6, 3,
                         9, -9, 1, 8,  -8, 1, -3, 5, -3, -6, -1, 2, -3};
  solveQp(lp);

  lp = HighsLp();
  lp.num_col_ = 12;
  lp.num_row_ = 8;
  lp.row_lower_ = {-658690,  -1937588, -1304117, -1033898,
                   -8072220, 5707108,  -3619694, -9796327};
  lp.col_lower_ = {0, -inf, 0, -inf, 0, 0, -inf, -inf, 0, 0, -inf, 0};
  lp.col_upper_.assign(lp.num_col_, inf);
  lp.a_matrix_.start_ = {0, 3, 7, 8, 10, 13, 14, 18, 20, 21, 21, 25, 26};
  lp.a_matrix_.index_ = {0, 1, 4, 0, 2, 5, 7, 7, 2, 3, 1, 6, 7,
                         4, 2, 3, 4, 5, 3, 7, 7, 1, 5, 6, 7, 4};
  lp.a_matrix_.value_ = {-3, -9, -9, 5,  -8, -2, 5,  -7, -9, 4, 4,  -2, -9,
                         -3, 8,  1,  -5, -9, -5, -9, -5, -9, 9, -8, -7, -7};
  solveQp(lp);
}

TEST_CASE("test-hipo-infeasibility-certificate", "[highs_hipo]") {
  // The certificates that HiPO requires before it declares infeasibility
  // must reject an iterate whose objective or residuals overflow
  //
  // min -x1 - x2  s.t.  x1 - x2 = 1,  x1 - x2 + x3 = 1,  x >= 0
  // is unbounded, with ray (1, 1, 0).
  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 2;
  lp.col_cost_ = {-1, -1, 0};
  lp.col_lower_ = {0, 0, 0};
  lp.col_upper_ = {kHighsInf, kHighsInf, kHighsInf};
  lp.row_lower_ = {1, 1};
  lp.row_upper_ = {1, 1};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 2, 4, 5};
  lp.a_matrix_.index_ = {0, 1, 0, 1, 1};
  lp.a_matrix_.value_ = {1, 1, -1, -1, 1};

  hipo::Model model;
  REQUIRE(model.init(lp, HighsHessian()) == 0);
  REQUIRE(model.n() == 3);
  REQUIRE(model.m() == 2);
  hipo::Info info;
  hipo::Regularisation regul;
  hipo::Iterate it(model, info, regul);
  const double scale_x = model.colScale(0);
  REQUIRE(model.colScale(1) == scale_x);

  // A scaled ray is a certificate of dual infeasibility, not of primal
  // infeasibility
  it.x = {1e6 / scale_x, 1e6 / scale_x, 0};
  it.y = {0, 0};
  it.zl = {1, 1, 1};
  it.zu = {0, 0, 0};
  REQUIRE(it.dualInfeasCertificate() >= hipo::kInfeasCertificateTol);
  REQUIRE(it.primalInfeasCertificate() == 0);

  // Near the largest double, c^T x overflows to -inf: no certificate
  auto dot = [](const std::vector<double>& u, const std::vector<double>& v) {
    double s = 0;
    for (size_t i = 0; i < u.size(); ++i) s += u[i] * v[i];
    return s;
  };
  it.x = {1e308, 1e308, 1e308};
  REQUIRE(std::isinf(dot(it.x, model.c())));
  REQUIRE(it.dualInfeasCertificate() == 0);
  it.x = {kHighsInf, kHighsInf, 0};
  REQUIRE(it.dualInfeasCertificate() == 0);

  // b^T y overflows while A^T y + z is not zero: no certificate
  it.x = {1, 1, 1};
  it.y = {1e308, 1e308};
  REQUIRE(std::isinf(dot(it.y, model.b())));
  REQUIRE(it.primalInfeasCertificate() == 0);
  it.y = {kHighsInf, 0};
  REQUIRE(it.primalInfeasCertificate() == 0);
}

TEST_CASE("test-hipo-freevar", "[highs_hipo]") {
  const HighsModelStatus expected_status = HighsModelStatus::kOptimal;
  Highs highs;
  runHipoTest(highs, "perold.mps", -9.381e3, expected_status, "on");
  runHipoTest(highs, "perold.mps", -9.381e3, expected_status, "off");
  highs.resetThreadScheduler(true);
}

// min c'x s.t. Ax >= b with free x, random entries +-1..9, built around a
// point x0 with entries up to 1e7 (c = A'y for some y >= 0, so the LP is
// bounded). Integer data from splitmix64, so the LP is the same everywhere.
static HighsLp largeFreeVarsLp(const HighsInt num_col, const HighsInt num_row,
                               const int64_t scale, uint64_t seed) {
  auto next = [&seed]() {
    uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  const HighsInt per_row = 4;
  std::vector<int64_t> x0(num_col);
  for (auto& v : x0) v = int64_t(next() % uint64_t(2 * scale + 1)) - scale;
  std::vector<std::vector<std::pair<HighsInt, int64_t>>> rows(num_row);
  for (auto& row : rows) {
    while (HighsInt(row.size()) < per_row) {
      const HighsInt j = next() % num_col;
      bool found = false;
      for (const auto& entry : row) found = found || entry.first == j;
      if (!found) row.push_back({j, 0});
    }
    for (auto& entry : row) {
      entry.second = 1 + int64_t(next() % 9);
      if (next() % 2) entry.second = -entry.second;
    }
  }
  HighsLp lp;
  lp.num_col_ = num_col;
  lp.num_row_ = num_row;
  std::vector<int64_t> cost(num_col, 0);
  std::vector<std::vector<std::pair<HighsInt, int64_t>>> cols(num_col);
  for (HighsInt i = 0; i < num_row; i++) {
    int64_t y = 0, s = 0, ax = 0;
    if (next() % 2) y = next() % 10;
    if (next() % 2) s = next() % uint64_t(scale / 10);
    for (const auto& entry : rows[i]) {
      cost[entry.first] += entry.second * y;
      ax += entry.second * x0[entry.first];
      cols[entry.first].push_back({i, entry.second});
    }
    lp.row_lower_.push_back(double(ax - s));
    lp.row_upper_.push_back(kHighsInf);
  }
  for (int64_t c : cost) lp.col_cost_.push_back(double(c));
  lp.col_lower_.assign(num_col, -kHighsInf);
  lp.col_upper_.assign(num_col, kHighsInf);
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0};
  for (const auto& col : cols) {
    for (const auto& entry : col) {
      lp.a_matrix_.index_.push_back(entry.first);
      lp.a_matrix_.value_.push_back(double(entry.second));
    }
    lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
  }
  return lp;
}

TEST_CASE("test-hipo-large-free-vars", "[highs_hipo]") {
  // Free variables far beyond their initial artificial bounds of +-1e4: HiPO
  // used to move one bound at a time to 2|x|, at almost every iteration, and
  // took 60 iterations here (21 with bounds moved to +-10|x|; 22 with bounds
  // +-1e9 instead of free columns)
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("presolve", kHighsOffString);
  REQUIRE(highs.passModel(largeFreeVarsLp(500, 1000, 10000000, 1)) ==
          HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(highs.getInfo().ipm_iteration_count <= 35);
  highs.resetThreadScheduler(true);
}

TEST_CASE("test-hipo-zero-cost-dependent-rows", "[highs_hipo]") {
  // min 0 s.t. -x0 - 2x1 + 4x2 >= -1201505, x0 + 2x1 - 4x2 = 1201505,
  // x0 >= 0, x1, x2 free. The rows are dependent, so y can grow along the
  // ray y0 = y1 of the optimal dual face, and the rounding errors in b'y
  // keep the gap above the tolerance while mu goes to zero. With the
  // artificial bounds of x1, x2 moved to +-10|x|, HiPO took 294 iterations to
  // stop, and IPX then dropped all barrier terms and iterated without end,
  // as a zero complementarity gap that stays zero did not count as a stall.
  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 2;
  lp.col_cost_ = {0, 0, 0};
  lp.col_lower_ = {0, -kHighsInf, -kHighsInf};
  lp.col_upper_ = {kHighsInf, kHighsInf, kHighsInf};
  lp.row_lower_ = {-1201505, 1201505};
  lp.row_upper_ = {kHighsInf, 1201505};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 2, 4, 6};
  lp.a_matrix_.index_ = {0, 1, 0, 1, 0, 1};
  lp.a_matrix_.value_ = {-1, 1, -2, 2, 4, -4};
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("time_limit", 10.0);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(highs.getInfo().ipm_iteration_count <= 60);
  highs.resetThreadScheduler(true);
}

TEST_CASE("test-hipo-huge-finite-bound", "[highs_hipo]") {
  // min x0 + x1 + x2 s.t. x0 + x1 + x2 >= 1, 0 <= x0 <= 1e200, x1 free,
  // x2 >= 0, with infinite_bound = inf so that 1e200 is a finite bound. The
  // iterate is of order 1e198, and moving the artificial bounds of x1 makes
  // the products at the trial point of the centrality corrector overflow.
  // The corrector must be skipped rather than end the solve with NaN.
  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 1;
  lp.col_cost_ = {1, 1, 1};
  lp.col_lower_ = {0, -kHighsInf, 0};
  lp.col_upper_ = {1e200, kHighsInf, kHighsInf};
  lp.row_lower_ = {1};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2, 3};
  lp.a_matrix_.index_ = {0, 0, 0};
  lp.a_matrix_.value_ = {1, 1, 1};
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("infinite_bound", kHighsInf);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("time_limit", 10.0);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(std::abs(highs.getInfo().objective_function_value - 1) < 1e-6);
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

TEST_CASE("test-hipo-inf-iterate", "[highs_hipo]") {
  // min   1e19 x + y
  // s.t.  x + y >= 1
  //       0 <= x <= 1e300, y free
  //
  // With infinite_bound = inf, the bound 1e300 is finite. In the starting
  // point, mu = sum xu * zu overflows and the iterate becomes infinite at
  // iteration 0. HiPO used to carry on, switch to FactorHighs after the NaN
  // direction and reset to a best iterate that was never saved, which
  // crashed in Model::resetBounds() because y is free.

  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {1e19, 1};
  lp.col_lower_ = {0, -kHighsInf};
  lp.col_upper_ = {1e300, kHighsInf};
  lp.row_lower_ = {1};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;

  Highs h;
  h.setOptionValue("output_flag", dev_run);
  h.setOptionValue("infinite_bound", kHighsInf);
  REQUIRE(h.passModel(lp) == HighsStatus::kOk);
  h.setOptionValue("solver", "hipo");
  h.setOptionValue("presolve", "off");
  h.run();
  const HighsModelStatus model_status = h.getModelStatus();
  REQUIRE((model_status == HighsModelStatus::kOptimal ||
           model_status == HighsModelStatus::kSolveError));
  h.resetThreadScheduler(true);
}

TEST_CASE("test-hipo-no-best-iterate", "[highs_hipo]") {
  // min   1000 x1 + y + 1e-6 x3
  // s.t.  x1 + y + 1e-8 x3 >= 1
  //       0 <= x1 <= 1e306, y free, x3 >= 0
  //
  // With infinite_bound = inf, the starting point is finite but its primal
  // infeasibility overflows (column scaling of the bound 1e306), so no best
  // iterate is saved at iteration 0. The direction at iteration 1 is NaN,
  // so HiPO switches to FactorHighs and resets to the best iterate; that
  // reset used to copy the empty best iterate and crash in
  // Model::resetBounds() because y is free. The optimum is y = 1 with
  // objective 1.

  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 1;
  lp.col_cost_ = {1000, 1, 1e-6};
  lp.col_lower_ = {0, -kHighsInf, 0};
  lp.col_upper_ = {1e306, kHighsInf, kHighsInf};
  lp.row_lower_ = {1};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.start_ = {0, 1, 2, 3};
  lp.a_matrix_.index_ = {0, 0, 0};
  lp.a_matrix_.value_ = {1, 1, 1e-8};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;

  Highs h;
  h.setOptionValue("output_flag", dev_run);
  h.setOptionValue("infinite_bound", kHighsInf);
  REQUIRE(h.passModel(lp) == HighsStatus::kOk);
  h.setOptionValue("solver", "hipo");
  h.setOptionValue("presolve", "off");
  h.run();
  const HighsModelStatus model_status = h.getModelStatus();
  REQUIRE((model_status == HighsModelStatus::kOptimal ||
           model_status == HighsModelStatus::kSolveError));
  if (model_status == HighsModelStatus::kOptimal)
    REQUIRE(std::fabs(h.getInfo().objective_function_value - 1) < 1e-6);
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
