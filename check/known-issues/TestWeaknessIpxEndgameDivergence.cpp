// Unreported (found in a code review of latest)
// IPX stops with "no progress" on the second LP of a lexicographic pair,
// whose objective-bound row is tight on the whole feasible set
//
// The model is a multi-commodity network flow LP with side constraints:
// per commodity, flow conservation rows on a time-expanded network and a
// supply row, and one row (= 1) per task over the commodities allowed for
// it. Level 1 minimises cost A; level 2 adds the row cost_A'x <= opt_1 and
// minimises cost B. With opt_1 the level-1 optimum, that row is in effect
// the equality cost_A'x = opt_1, which also fixes every variable with a
// nonzero level-1 reduced cost at its bound: the LP has no strictly
// feasible point and its set of dual optima is unbounded.
//
// Symptom on latest (IPX, crossover off, presolve on), on this
// 9,064 x 12,612 LP: the bound row's dual grows from -44 to -376 in fifteen
// iterations while its slack falls to 1e-7, the steps shrink to 0.01-0.06,
// and IPX stops after five short steps at a gap of about 4e-3: "no
// progress", model status Unknown, P-D objective error 3e-3. Whether IPX
// gets through is erratic: with the bound changed by a relative 1e-13, it
// is Optimal for some values and Unknown for others. On 48 generated LPs
// (483 to 18,019 rows, exact level-1 bound) latest failed on 3, all with
// 9,000 rows or more. Around the exact bound (11 bounds within a relative
// 5e-13), IPX failed for 7 of 11 on this LP and 2-3 of 11 on two others of
// the same size, never on smaller ones. HiPO stalls on this LP for all 11
// (gap 4e-3) and restarts IPX, which mostly stops too: on the seven bounds
// of the tests below, latest ends Unknown for 5. The LP need not be exactly
// tight: with the bound a relative 1e-9 or 1e-7 above opt_1 (a thin but
// strictly feasible set) IPX and HiPO still end Unknown on this LP; at 1e-6
// above, IPX solves all three LPs (HiPO one of them). The fix for short
// rebalancing steps (branch fixes/ipx-rebalancing-short-steps) doesn't
// change the outcome.
//
// Cause, on these seven bounds: near the optimum, with the residuals within
// tolerance, IPX takes runs of short steps in which sigma is 0.5-0.85, so
// they are mostly centring; the complementarity gap stays put and the
// primal residual keeps falling. After a few steps the iterate is central
// again, and IPX takes long steps and converges. IPM::Driver stops after
// five short steps in a row, which cut off the runs of 6 steps that these
// bounds take (IPX), and of up to 19 steps in IPX's restart from HiPO's
// iterate. With that stop disabled, IPX is Optimal on all seven bounds
// after 65-69 iterations. On two larger generated LPs (S40), IPX crawls
// for about 40 iterations and then really diverges (the dual residual
// grows from 1e-11 to 3e-9): there the earlier explanation still seems to
// apply, that the inexact (CR) Newton directions lose the accuracy the IPM
// needs. Solving the KKT systems 100 times more accurately (kkt_tol 0.003)
// made IPX solve all three LPs then tested, but it then failed on another
// LP of the 48 and doubled the CR iterations; tightening it only after
// short steps did not help; larger ipm_drop_primal/ipm_drop_dual did not
// help.
//
// Note: the generator that found this wrote opt_1 with six significant
// digits. Rounded down, the bound makes the LP infeasible by about 1e-5
// (simplex: Infeasible), and then IPX's dual objective runs away along the
// Farkas ray, also "no progress", while the primal residual stays at 1e-9.
// That is an artefact of the rounding, not this weakness.
//
// Fix: fixes/ipx-endgame-recentring. The main IPM tolerates a run of short
// steps while it is recentring: the iterate is within the feasibility
// tolerances and |pobj-dobj| stays within 10 times its size at the start
// of the run (in a real stall, e.g. with the bound lowered so that the LP
// is infeasible, it grows by about 3x per step), for at most 25 steps.
// IPX and HiPO (through its IPX restart) are then Optimal on all seven
// bounds. fixes/3724-hipo-predictor-sigma alone doesn't help: HiPO still
// stalls and restarts IPX on all seven bounds, and the solve ends Optimal
// for 3 of them (2 on latest). With one bound, the HiPO case passed with
// that fix by chance, so it now loops over the seven bounds, like the IPX
// case.
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

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

namespace {
// Optimum of level 1 (cost A) on the same network, by simplex
const double kOpt1 = 325.6950297420795;

HighsModelStatus solveLexLevel2(const std::string& solver, const double bound,
                                LogCapture& log) {
  Highs highs;
  captureLog(highs, log);
  highs.setOptionValue("solver", solver);
  highs.setOptionValue("run_crossover", kHighsOffString);
  REQUIRE(highs.passModel(lexLevel2(20, 3, 6, 5, 5, bound)) ==
          HighsStatus::kOk);
  REQUIRE(highs.getLp().num_row_ == 9064);
  REQUIRE(highs.getLp().num_col_ == 12612);
  highs.run();
  return highs.getModelStatus();
}
}  // namespace

TEST_CASE("weakness-ipx-endgame-divergence-lex-level2",
          "[known-issue][!shouldfail]") {
  // The outcome is erratic in the last digits of the bound, so this solves
  // the LP for seven bounds within a relative 3e-13 of the level-1 optimum
  HighsInt num_not_optimal = 0;
  std::ostringstream outcomes;  // one message: each assertion clears INFO
  for (HighsInt k = -3; k <= 3; k++) {
    const double bound = kOpt1 * (1 + k * 1e-13);
    LogCapture log;
    const HighsModelStatus status = solveLexLevel2(kIpxString, bound, log);
    outcomes << "\nbound " << std::setprecision(17) << bound << ": "
             << Highs().modelStatusToString(status);
    if (status != HighsModelStatus::kOptimal) num_not_optimal++;
  }
  INFO(outcomes.str());
  // latest: IPX stops with "no progress" (Unknown) for 3 of the 7 bounds
  REQUIRE(num_not_optimal == 0);
}

#ifdef HIPO
TEST_CASE("weakness-ipx-endgame-divergence-lex-level2-hipo",
          "[known-issue][!shouldfail]") {
  // As for IPX, the outcome is erratic in the last digits of the bound
  HighsInt num_not_optimal = 0;
  std::ostringstream outcomes;
  for (HighsInt k = -3; k <= 3; k++) {
    const double bound = kOpt1 * (1 + k * 1e-13);
    LogCapture log;
    const HighsModelStatus status = solveLexLevel2(kHipoString, bound, log);
    REQUIRE(log.contains("HiPO iterations"));
    outcomes << "\nbound " << std::setprecision(17) << bound << ": "
             << Highs().modelStatusToString(status);
    if (status != HighsModelStatus::kOptimal) num_not_optimal++;
  }
  INFO(outcomes.str());
  // latest: HiPO stalls and restarts IPX for all 7 bounds, and the solve
  // ends Unknown for 5 of them
  REQUIRE(num_not_optimal == 0);
}
#endif
