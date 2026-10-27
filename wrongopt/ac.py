"""The analytic centre as HighsMipSolverData::startAnalyticCenterComputation computes it (presolved MIP
model, zero cost, IPX, no presolve, no crossover, centring, 200 iterations), and the columns that
finishAnalyticCenterComputation would fix at a bound. Column 35545 (presolved) is the one whose fixing
cuts off the 56 solution in the Xcode 15.4 build. With the IPX log, then without centring, and the
largest value column 35545 takes on the LP polytope.
usage: ac.py MODEL"""
import sys
import highspy
import numpy as np

COL = 35545
h = highspy.Highs()
h.setOptionValue("output_flag", False)
h.readModel(sys.argv[1])
h.presolve()
lp = h.getPresolvedLp()
print("presolved", lp.num_row_, "rows", lp.num_col_, "cols", flush=True)
lo, up = np.array(lp.col_lower_), np.array(lp.col_upper_)
lp.col_cost_ = [0.0] * lp.num_col_
lp.integrality_ = []


def centre(tag, log=False, **opts):
    ipm = highspy.Highs()
    ipm.setOptionValue("output_flag", log)
    ipm.setOptionValue("log_dev_level", 1 if log else 0)
    base = {"presolve": "off", "solver": "ipx", "ipm_iteration_limit": 200, "run_crossover": "off",
            "run_centring": True}
    for k, v in {**base, **opts}.items():
        assert ipm.setOptionValue(k, v) == highspy.HighsStatus.kOk, (k, v)
    ipm.passModel(lp)
    ipm.run()
    x = np.array(ipm.getSolution().col_value)
    rng = up - lo
    tol = 1e-6 * np.minimum(rng, 1.0)  # mip_feasibility_tolerance 1e-6
    free = rng > 0
    atlo = free & (x <= lo + tol)
    atup = free & ~atlo & (x >= up - tol)
    print(tag, ipm.modelStatusToString(ipm.getModelStatus()), "iterations", ipm.getInfo().ipm_iteration_count,
          f"x[{COL}] = {x[COL]:.3e}", "fixed at lower", int(atlo.sum()), "at upper", int(atup.sum()),
          "smallest distance to the lower bound above tolerance", f"{(x - lo)[free & ~atlo].min():.3e}", flush=True)
    d = (x - lo)[free & ~atlo]
    print(tag, "distances to the lower bound above tolerance by decade:",
          {f"1e{k}": int(((d >= 10.0 ** k) & (d < 10.0 ** (k + 1))).sum()) for k in range(-6, 0)}, flush=True)
    return x


x = centre("as in the MIP solver:", log=True)
x0 = centre("without centring:", run_centring=False)
print(f"max |difference| between the two: {np.abs(x - x0).max():.3e}", flush=True)
mx = highspy.Highs()
mx.setOptionValue("output_flag", False)
cost = [0.0] * lp.num_col_
cost[COL] = -1.0
lp.col_cost_ = cost
mx.passModel(lp)
mx.run()
print(f"max x[{COL}] on the LP polytope:", -mx.getInfo().objective_function_value, flush=True)
