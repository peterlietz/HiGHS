// Unreported (found in a code review of latest)
// IPX stops with "no progress" on the second LP of a lexicographic pair,
// whose objective-bound row is tight on the whole feasible set
//
// The model is a multi-fleet time-space network LP: per fleet, flow
// conservation rows at (station, event time), a supply row, and one
// coverage row (= 1) per flight over the fleets allowed to fly it. Level 1
// minimises cost A; level 2 adds the row cost_A'x <= opt_1 and minimises
// cost B. With opt_1 the level-1 optimum, that row holds with equality at
// every feasible point, so the LP has no strictly feasible point and its
// set of dual optima is unbounded.
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
// (gap 4e-3, then its IPX restart stops too). The LP need not be exactly
// tight: with the bound a relative 1e-9 or 1e-7 above opt_1 (a thin but
// strictly feasible set) IPX and HiPO still end Unknown on this LP; at 1e-6
// above, IPX solves all three LPs (HiPO one of them). The fix for short
// rebalancing steps (branch fixes/ipx-rebalancing-short-steps) doesn't
// change the outcome.
//
// Cause: with no (or a very thin) strictly feasible set, the central path
// doesn't exist (or hugs the bound row): the dual of the bound row
// diverges, the normal equations become ill-conditioned along that
// direction, and the inexact (CR) Newton directions apparently lose the
// accuracy the IPM needs (measured indirectly, see next sentence). Solving
// the KKT systems 100 times more accurately (kkt_tol 0.003) made IPX solve
// all three, but it then failed on another LP of the 48 and doubled the
// CR iterations; tightening it only after short steps did not help; larger
// ipm_drop_primal/ipm_drop_dual did not help.
//
// Note: the generator that found this wrote opt_1 with six significant
// digits. Rounded down, the bound makes the LP infeasible by about 1e-5
// (simplex: Infeasible), and then IPX's dual objective runs away along the
// Farkas ray, also "no progress", while the primal residual stays at 1e-9.
// That is an artefact of the rounding, not this weakness.
//
// Fix: none yet
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
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

// Level 2 of a lexicographic pair on a time-space network with
// num_station stations, num_day days and num_fleet fleets of num_aircraft
// to 2 * num_aircraft - 1 aircraft each: minimise cost B subject to
// cost_A'x <= level1_bound. Every fleet may fly every flight; 80% of the
// flight arcs have cost A 1, 5% have cost B 1; all arcs have upper bound 79.
HighsLp lexLevel2(const HighsInt num_station, const HighsInt num_day,
                  const HighsInt num_fleet, const HighsInt num_aircraft,
                  const uint64_t seed, const double level1_bound) {
  const double kArcUpper = 79;
  const HighsInt kCostAPct = 80;
  const HighsInt kExtraFleetPct = 100;
  const HighsInt kCostBPct = 5;
  SplitMix64 rng{seed};
  std::vector<HighsInt> supply;
  for (HighsInt f = 0; f < num_fleet; f++)
    supply.push_back(num_aircraft + rng.below(num_aircraft));
  const HighsInt horizon = num_day * 1440;
  struct Flight {
    HighsInt orig, dest, dep, ready, fleet;
  };
  // Flights come from generated rotations, so the LP is feasible
  std::vector<Flight> flights;
  for (HighsInt f = 0; f < num_fleet; f++) {
    for (HighsInt k = 0; k < supply[f]; k++) {
      HighsInt s = rng.below(num_station);
      HighsInt t = rng.below(120);
      while (true) {
        HighsInt d = rng.below(num_station - 1);
        if (d >= s) d++;
        const HighsInt dur = 45 + rng.below(240);
        if (t + dur + 30 >= horizon) break;
        flights.push_back({s, d, t, t + dur + 30, f});
        s = d;
        t = t + dur + 30 + rng.below(180);
      }
    }
  }
  std::vector<std::vector<HighsInt>> allowed;
  for (const Flight& fl : flights) {
    std::set<HighsInt> al = {fl.fleet};
    for (HighsInt g = 0; g < num_fleet; g++)
      if (g != fl.fleet && rng.below(100) < kExtraFleetPct) al.insert(g);
    allowed.emplace_back(al.begin(), al.end());
  }
  // Event times per (fleet, station)
  std::vector<std::set<HighsInt>> events(num_fleet * num_station,
                                         std::set<HighsInt>{0, horizon});
  for (size_t i = 0; i < flights.size(); i++) {
    for (HighsInt f : allowed[i]) {
      events[f * num_station + flights[i].orig].insert(flights[i].dep);
      events[f * num_station + flights[i].dest].insert(flights[i].ready);
    }
  }
  HighsLp lp;
  // Rows: supply per fleet, node rows, coverage rows, the level-1 bound
  for (HighsInt f = 0; f < num_fleet; f++) {
    lp.row_lower_.push_back(supply[f]);
    lp.row_upper_.push_back(supply[f]);
  }
  std::vector<std::map<HighsInt, HighsInt>> node_row(num_fleet * num_station);
  for (HighsInt fs = 0; fs < num_fleet * num_station; fs++) {
    for (HighsInt t : events[fs]) {
      node_row[fs][t] = lp.row_lower_.size();
      lp.row_lower_.push_back(0);
      lp.row_upper_.push_back(0);
    }
  }
  const HighsInt first_cover_row = lp.row_lower_.size();
  for (size_t i = 0; i < flights.size(); i++) {
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
  for (HighsInt f = 0; f < num_fleet; f++) {
    for (HighsInt s = 0; s < num_station; s++) {
      const std::map<HighsInt, HighsInt>& rows = node_row[f * num_station + s];
      addColumn(0, {{f, 1}, {rows.at(0), 1}});
      for (auto it = rows.begin(); std::next(it) != rows.end(); ++it)
        addColumn(0, {{it->second, -1}, {std::next(it)->second, 1}});
      addColumn(0, {{rows.at(horizon), -1}});
    }
  }
  // Flight arcs
  for (size_t i = 0; i < flights.size(); i++) {
    const Flight& fl = flights[i];
    for (HighsInt f : allowed[i]) {
      const bool cost_a = rng.below(100) < kCostAPct;
      const bool cost_b = rng.below(100) < kCostBPct;
      std::vector<std::pair<HighsInt, double>> ents = {
          {node_row[f * num_station + fl.orig].at(fl.dep), -1},
          {node_row[f * num_station + fl.dest].at(fl.ready), 1},
          {first_cover_row + HighsInt(i), 1}};
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
  for (HighsInt k = -3; k <= 3; k++) {
    const double bound = kOpt1 * (1 + k * 1e-13);
    LogCapture log;
    const HighsModelStatus status = solveLexLevel2(kIpxString, bound, log);
    UNSCOPED_INFO("bound " << std::setprecision(17) << bound << ": "
                           << Highs().modelStatusToString(status));
    if (status != HighsModelStatus::kOptimal) num_not_optimal++;
  }
  // latest: IPX stops with "no progress" (Unknown) for 4 of the 7 bounds
  REQUIRE(num_not_optimal == 0);
}

#ifdef HIPO
TEST_CASE("weakness-ipx-endgame-divergence-lex-level2-hipo",
          "[known-issue][!shouldfail]") {
  LogCapture log;
  const HighsModelStatus status = solveLexLevel2(kHipoString, kOpt1, log);
  REQUIRE(log.contains("HiPO iterations"));
  // latest: HiPO stalls at a gap of 4e-3 and restarts IPX, which stops with
  // "no progress": Unknown
  REQUIRE(status == HighsModelStatus::kOptimal);
}
#endif
