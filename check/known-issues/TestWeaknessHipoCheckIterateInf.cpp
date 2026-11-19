// ERGO-Code/HiGHS#3748 (open)
// HiPO doesn't stop on an infinite iterate: checkIterate() declares a new
// local `bool terminate = true;` in its isInf() branch
//
// The LP: min 1e19 x + y s.t. x + y >= 1, 0 <= x <= 1e300, y >= 0, with
// infinite_bound = inf so that 1e300 stays a finite bound (optimum x = 0,
// y = 1, objective 1; simplex and IPX solve it). In HiPO's starting point,
// xu ~ 1e300 and zu ~ 1e19, so mu = sum xu * zu overflows and the
// centrality shift makes xl and xu infinite at iteration 0.
//
// Symptom on latest: "Iterate is inf" sets info_.error but prepareIter()
// carries on. The predictor factorises with infinite scaling, the direction
// is NaN, HiPO switches to FactorHighs ("Switching to FactorHighs because of
// bad direction") and resets to its best iterate, which was never saved
// (best_iter = -1: the violation at iteration 0 is inf/NaN), so the iterate
// becomes empty vectors ("Resetting to iteration -1") and computeMu() and
// residual1234() read past their ends. It ends with "Direction is nan":
// Solve error after 1 iteration and 4 solves, instead of stopping at
// iteration 0 after 2 solves. If the LP has a free variable (y free
// instead of y >= 0), Model::resetBounds() then dereferences the empty
// best_lower_ and HiGHS segfaults (exit 139 from the highs binary on
// latest); that variant is not run here because it would kill unit_tests.
// The same happens with cost 1e300, bound 1e19 and infinite_cost = inf.
//
// Cause: the shadowing in Solver::checkIterate(), so checkIterate() returns
// false for an infinite iterate; checkBadIter() and checkTermination() then
// run on it and the error is only seen in the next solveNewtonSystem().
//
// Fix: branch fixes/hipo-checkiterate-inf (assigns the outer `terminate`;
// HiPO then stops at iteration 0 with "NaN detected", Solve error, as for a
// NaN iterate).
//
// Test design: letting latest run on into the reads past the end of the
// empty vectors corrupts the heap, and a later test in the same unit_tests
// process then crashes (SIGTRAP from the allocator, in 2 of 5 runs). So the
// test interrupts the solve at HiPO's first interrupt check after "Iterate
// is inf", through kCallbackIpmInterrupt, when that message is still the
// last one in the log. On latest that check comes next, in prepareIter(),
// before the predictor; with the fix, HiPO has stopped and logged more
// before any further interrupt check, so no interrupt is requested.
#include <string>

#include "HConfig.h"
#include "KnownIssues.h"
#include "catch.hpp"

#ifdef HIPO
using namespace known_issues;

TEST_CASE("weakness-hipo-checkiterate-inf-starting-point",
          "[known-issue][!shouldfail]") {
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {1e19, 1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1e300, kHighsInf};
  lp.row_lower_ = {1};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};

  struct State {
    std::string log;
    bool interrupted_after_inf = false;
  } state;
  Highs highs;
  highs.setOptionValue("output_flag", true);
  highs.setOptionValue("log_to_console", kDevRun);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("infinite_bound", kHighsInf);
  // HiPO's "Iterate is inf" message is info level
  highs.setOptionValue("log_dev_level", 1);
  highs.setCallback(
      [](int callback_type, const std::string& message,
         const HighsCallbackOutput*, HighsCallbackInput* data_in,
         void* user_data) {
        auto& state = *static_cast<State*>(user_data);
        if (callback_type == kCallbackLogging) state.log += message;
        if (callback_type == kCallbackIpmInterrupt) {
          const std::string inf = "Iterate is inf\n";
          const bool inf_is_last =
              state.log.size() >= inf.size() &&
              state.log.compare(state.log.size() - inf.size(), inf.size(),
                                inf) == 0;
          if (inf_is_last) {
            state.interrupted_after_inf = true;
            data_in->user_interrupt = true;
          }
        }
      },
      &state);
  highs.startCallback(kCallbackLogging);
  highs.startCallback(kCallbackIpmInterrupt);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  highs.run();
  REQUIRE(state.log.find("Iterate is inf") != std::string::npos);
  // latest: HiPO carries on from the infinite iterate to its interrupt check
  REQUIRE_FALSE(state.interrupted_after_inf);
}
#endif
