// Unreported (found in a code review of latest)
// solver = "ipm" runs HiPO even when its Cholesky factor is nearly dense, on
// network LPs that IPX solves many times faster
//
// The model is a random min-cost flow: 6000 nodes, a cycle of uncapacitated
// arcs with cost 1000 (so that it is feasible), 30000 random arcs with costs
// 1-100 and capacities 1-50, and 600 random supply/demand pairs. HiPO's
// analyse phase (NE system) predicts a factor with 5.9e6 nonzeros (half of a
// dense 6000 x 6000 triangle) and 1.3e10 flops per factorisation, which is
// 1.8e5 flops per nonzero of A.
//
// Symptom on latest (macOS arm64, 10 threads, run_crossover off, timings
// repeated on a loaded machine): solver = "ipm" (HiPO) takes 5.3-5.4 s in
// 18 iterations, solver = "ipx" takes 0.8-1.2 s in 23 iterations. Mittelmann's
// network LPs are worse: wide15, long15 and square15 (6.8e12 flops per
// factorisation) and netlarge2 (1.1e13) hit a 300 s time limit after 4-9
// HiPO iterations where IPX needs 15-35 s, and lo10 (9.9e11) takes 131-200 s
// against IPX's 11-15 s.
//
// Cause: useHipo() picks HiPO for solver = "ipm" whenever HiPO is available
// ("Later decide between HiPO and IPX based on LP properties"). For a
// network matrix every basis is triangular, so IPX's basis preconditioner is
// cheap, whereas the Cholesky factor of A * D * A^T of a graph without small
// separators is nearly dense.
//
// Fix: branch fixes/hipo-dense-fill. After its analyse phase, HiPO leaves a
// network LP (at most two nonzeros per column) to IPX when the predicted
// factorisation needs more than 1e5 flops per nonzero of A.
#include <cstdint>

#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

#ifdef HIPO
using namespace known_issues;

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
};

// Random min-cost flow on num_node nodes with degree * num_node random arcs
HighsLp randomNetwork(const HighsInt num_node, const HighsInt degree,
                      const uint64_t seed) {
  const HighsInt m = num_node;
  std::vector<HighsInt> tail, head;
  std::vector<double> cost, upper;
  for (HighsInt i = 0; i < m; i++) {
    tail.push_back(i);
    head.push_back((i + 1) % m);
    cost.push_back(1000);
    upper.push_back(kHighsInf);
  }
  SplitMix64 rng{seed};
  for (HighsInt k = 0; k < degree * m; k++) {
    const HighsInt i = rng.next() % m;
    HighsInt j = rng.next() % m;
    if (i == j) j = (j + 1) % m;
    tail.push_back(i);
    head.push_back(j);
    cost.push_back(double(1 + rng.next() % 100));
    upper.push_back(double(1 + rng.next() % 50));
  }
  std::vector<double> supply(m, 0);
  for (HighsInt k = 0; k < m / 10; k++) {
    const HighsInt i = rng.next() % m;
    const HighsInt j = rng.next() % m;
    const double v = double(1 + rng.next() % 100);
    supply[i] += v;
    supply[j] -= v;
  }
  HighsLp lp;
  lp.num_col_ = tail.size();
  lp.num_row_ = m;
  lp.col_cost_ = cost;
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_ = upper;
  lp.row_lower_.resize(m);
  for (HighsInt i = 0; i < m; i++) lp.row_lower_[i] = -supply[i];
  lp.row_upper_ = lp.row_lower_;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0};
  for (HighsInt j = 0; j < lp.num_col_; j++) {
    lp.a_matrix_.index_.push_back(tail[j]);
    lp.a_matrix_.value_.push_back(-1);
    lp.a_matrix_.index_.push_back(head[j]);
    lp.a_matrix_.value_.push_back(1);
    lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
  }
  return lp;
}
}  // namespace

TEST_CASE("weakness-hipo-dense-fill-network", "[known-issue][!shouldfail]") {
  Highs highs;
  LogCapture log;
  captureLog(highs, log);
  highs.setOptionValue("solver", kIpmString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  REQUIRE(highs.passModel(randomNetwork(6000, 5, 1)) == HighsStatus::kOk);
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  // latest: solver = "ipm" solves the LP with HiPO (18 iterations, 5 s)
  // rather than with IPX (1 s)
  REQUIRE(log.contains("IPX model has"));
  REQUIRE_FALSE(log.contains("HiPO iterations"));
}
#endif
