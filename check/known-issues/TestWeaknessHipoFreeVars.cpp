// Unreported (found in a code review of latest)
// HiPO's artificial bounds on free variables: moving one bound to 2|x|
// stalls the IPM when free variables are large at the optimum
//
// The model is a random LP min c'x s.t. Ax >= b with 1000 free columns,
// 2000 rows, 4 entries +-1..9 per row, built around a point x0 with entries
// up to 1e7 (c = A'y for some y >= 0, so the LP is bounded).
//
// Symptom on latest (presolve off): HiPO stops after 68 iterations with
// five short steps in a row and restarts IPX ("Restarting with IPX"), which
// makes no progress either: Unknown after 73 IPM iterations. The same LP
// with bounds +-1e9 instead of free columns takes 23 HiPO iterations, and
// IPX takes 21 on the free LP. Over 27 such LPs (500-2000 columns, x0 up to
// 1e6-1e8), latest needed 1328 HiPO iterations in total, restarted IPX on 4
// and ended Unknown on 3; with the fix 578, no restart, all optimal (one
// within the relaxed tolerances).
//
// Cause: HiPO gives a free column the artificial bounds [-1e4, 1e4] and,
// when x gets within half of one of them, moves that bound to 2x
// (Model::adjustFreeVars). This leaves x exactly at the threshold, so a
// growing variable moves its bound at every iteration (on a 200-column
// instance: 1895 moves, up to 47 for one column); the other bound stays at
// +-1e4, so the box is centred near x and its barrier does not hold x back;
// and the residuals and mu are not recomputed after the move, so the next
// direction corrects bound residuals that no longer exist.
//
// Fix: branch fixes/hipo-free-vars
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

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

// min c'x s.t. Ax >= b, x free; feasible at x0 (|x0| <= scale) and bounded,
// since c = A'y with y >= 0. All data are integers.
HighsLp largeFreeVariables(const HighsInt num_col, const HighsInt num_row,
                           const HighsInt per_row, const int64_t scale,
                           const uint64_t seed) {
  SplitMix64 rng{seed};
  std::vector<int64_t> x0(num_col);
  for (auto& v : x0) v = int64_t(rng.next() % uint64_t(2 * scale + 1)) - scale;
  std::vector<std::vector<std::pair<HighsInt, int64_t>>> rows(num_row);
  for (auto& row : rows) {
    std::vector<HighsInt> cols;
    while (HighsInt(cols.size()) < per_row) {
      const HighsInt j = rng.next() % num_col;
      if (std::find(cols.begin(), cols.end(), j) == cols.end())
        cols.push_back(j);
    }
    for (HighsInt j : cols) {
      int64_t v = 1 + int64_t(rng.next() % 9);
      if (rng.next() % 2) v = -v;
      row.push_back({j, v});
    }
  }
  std::vector<int64_t> cost(num_col, 0);
  HighsLp lp;
  lp.num_col_ = num_col;
  lp.num_row_ = num_row;
  for (const auto& row : rows) {
    int64_t y = 0, s = 0;
    if (rng.next() % 2) y = rng.next() % 10;
    if (rng.next() % 2) s = rng.next() % uint64_t(scale / 10);
    int64_t ax = 0;
    for (const auto& entry : row) {
      cost[entry.first] += entry.second * y;
      ax += entry.second * x0[entry.first];
    }
    lp.row_lower_.push_back(double(ax - s));
    lp.row_upper_.push_back(kHighsInf);
  }
  for (int64_t c : cost) lp.col_cost_.push_back(double(c));
  lp.col_lower_.assign(num_col, -kHighsInf);
  lp.col_upper_.assign(num_col, kHighsInf);
  // row-wise entries to column-wise, rows in increasing order per column
  std::vector<std::vector<std::pair<HighsInt, int64_t>>> cols(num_col);
  for (HighsInt i = 0; i < num_row; i++)
    for (const auto& entry : rows[i])
      cols[entry.first].push_back({i, entry.second});
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
}  // namespace

TEST_CASE("weakness-hipo-free-vars-large-values",
          "[known-issue][!shouldfail]") {
  Highs highs;
  LogCapture log;
  captureLog(highs, log);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("presolve", kHighsOffString);
  REQUIRE(highs.passModel(largeFreeVariables(1000, 2000, 4, 10000000, 6)) ==
          HighsStatus::kOk);
  // latest: returns a warning, with model status Unknown
  highs.run();
  // latest: HiPO stops after 68 iterations and restarts IPX
  REQUIRE_FALSE(log.contains("Restarting with IPX"));
  // latest: Unknown
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  // fix: 21 iterations, as with bounds +-1e9 instead of free columns (23)
  REQUIRE(highs.getInfo().ipm_iteration_count <= 40);
}
#endif
