"""Solve model.mps with highspy as the engine poses it: mip_rel_gap 0, IPM for the MIP's LPs, no start.
A feasible solution of objective 56 exists, so 'Optimal' 58 is wrong.
usage: run.py MODEL [--debug-sol=FILE]  (FILE is checked against every bound change, cut and conflict;
the check needs a DEBUGSOL build with assertions enabled)"""
import sys
import highspy

args = dict(a[2:].split("=", 1) for a in sys.argv[2:])
h = highspy.Highs()
h.readModel(sys.argv[1])
h.setOptionValue("mip_rel_gap", 0.0)
h.setOptionValue("mip_lp_solver", "ipm")
if "debug-sol" in args:
    h.setOptionValue("mip_debug_solution_file", args["debug-sol"])
h.run()
info = h.getInfo()
print("RESULT", h.modelStatusToString(h.getModelStatus()), "objective", info.objective_function_value,
      "dual bound", info.mip_dual_bound, "nodes", info.mip_node_count, "seconds", round(h.getRunTime(), 1), flush=True)
