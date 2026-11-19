// ERGO-Code/HiGHS#3724 (open)
// HiPO stalls on the network LP 16_n14: the start is far outside the bounds
// and the predictor keeps sigma = 0.01
//
// 16_n14 (Mittelmann's network LPs, 16,384 x 262,144) is too big for a unit
// test, so this builds a smaller min-cost flow with the same structure: a
// backbone path through all nodes with cost 157 and capacity equal to the
// supply, plus random arcs with capacities 1-15. The supply, 39 per node,
// is spread by the least-squares starting point over all arcs, far outside
// the small capacities.
//
// Symptom on latest: HiPO's steps stay short, it stops after five steps
// below 0.05 and restarts IPX from its iterate ("Restarting with IPX"),
// which stalls too: Unknown. On 1024 and 2048 nodes with six seeds each,
// latest always needed the IPX restart, and ended Unknown for seeds 4 and 6
// on 2048 nodes; with the fix, all were Optimal in HiPO alone.
//
// Cause: the predictor always uses sigma = 0.01; the step-length-based sigma
// (up to 0.9 after short steps) is only used for the correctors. The
// combined direction keeps aiming at a much smaller mu and is blocked by the
// bounds, while pinf and mu fall together.
//
// Fix: branch fixes/3724-hipo-predictor-sigma (also the diff in the
// issue).
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

// Min-cost flow from node 0 to node N-1, shaped like 16_n14
HighsLp networkLikeN14(const HighsInt num_node, const HighsInt arcs_per_node,
                       const uint64_t seed) {
  const HighsInt supply = 39 * num_node;
  std::vector<HighsInt> tail, head;
  std::vector<double> cost, upper;
  for (HighsInt i = 0; i + 1 < num_node; i++) {
    tail.push_back(i);
    head.push_back(i + 1);
    cost.push_back(157);
    upper.push_back(supply);
  }
  SplitMix64 rng{seed};
  while (HighsInt(tail.size()) < num_node * arcs_per_node) {
    const HighsInt a = rng.next() % num_node;
    const HighsInt b = rng.next() % num_node;
    if (a == b) continue;
    const uint64_t c1 = rng.next() % 4096;
    const uint64_t c2 = rng.next() % 4096;
    const uint64_t c3 = rng.next() % 4096;
    tail.push_back(a);
    head.push_back(b);
    cost.push_back(double((c1 * c2 * c3) >> 24));
    upper.push_back(double(1 + rng.next() % 15));
  }
  HighsLp lp;
  lp.num_col_ = tail.size();
  lp.num_row_ = num_node;
  lp.col_cost_ = cost;
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_ = upper;
  lp.row_lower_.assign(num_node, 0);
  lp.row_lower_[0] = -supply;
  lp.row_lower_[num_node - 1] = supply;
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

TEST_CASE("issue-3724-hipo-predictor-sigma-network",
          "[known-issue][!shouldfail]") {
  Highs highs;
  LogCapture log;
  captureLog(highs, log);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("presolve", kHighsOffString);
  REQUIRE(highs.passModel(networkLikeN14(2048, 16, 4)) == HighsStatus::kOk);
  // latest: returns a warning, with model status Unknown
  highs.run();
  // latest: HiPO stops early and restarts IPX
  REQUIRE_FALSE(log.contains("Restarting with IPX"));
  // latest: Unknown
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
}
#endif
