# Known issues

Each test here reproduces a problem we found in HiGHS: an issue we reported
upstream (ERGO-Code/HiGHS) that is still open, or a weakness we found and
haven't reported. The tests carry the Catch2 tags
`[known-issue][!shouldfail]`:

- While the problem is present, the test fails, and Catch2 reports that as a
  pass, so `ctest` stays green.
- Once upstream fixes it, the test passes, and Catch2 reports that as a
  failure ("unexpected pass"). Then check the upstream change, keep the test
  as a regression test (tag `[known-issue-fixed]`, without `[!shouldfail]`)
  and drop or rework our fix branch. If upstream fixed only part of it, keep
  a `[!shouldfail]` case for the rest.

The branch `known-issues` holds one commit per problem, on top of `latest`.
Each commit only adds files in this directory, so a single problem's commit
can be dropped during a rebase without conflicts:

- `TestXxx.cpp`: the reproducer. Its header comment records the issue (or
  says that it is unreported), the symptom on `latest`, the cause, and the
  name of the branch with our fix, if there is one.
- `data/`: model files the reproducers read.

## Fixes

Our fixes are on branches named `fixes/<issue>-<topic>` or `fixes/<topic>`,
each on top of `latest`, or on top of the fix it depends on, so
`git diff latest fixes/<name>` is the fix. The test headers name the
branches; commit hashes change with every rebase, so nothing refers to
them. The branch `latest-with-fixes` is `known-issues` with every fix branch
merged in.

`fixes.txt` maps each case to the fix branches that make it pass on their
own. A case that passes only with several fixes together, or with none,
isn't listed.

## Running

The unit tests are only built with `-DALL_TESTS=ON` (the default `FAST_BUILD`
skips them). The HiPO tests also need `-DHIPO=ON`; without it they compile
to nothing.

```shell
cmake -S . -B build-hipo -DHIPO=ON -DALL_TESTS=ON && cmake --build build-hipo -j
build-hipo/bin/unit_tests "[known-issue]"                  # all known issues
build-hipo/bin/unit_tests "[known-issue-fixed]"            # regression tests of fixed ones
build-hipo/bin/unit_tests "[known-issue]" --list-tests     # what is tracked
build-hipo/bin/unit_tests "issue-3724*" -s                 # one, with details
```

A result of "All tests passed (N assertions in M test cases)" means that every
tracked problem is still present. A failing test case means that upstream
changed the behaviour; read its output.

`report.py` runs every case on its own and prints a table of what is still
present and what passes unexpectedly:

```shell
python3 -I check/known-issues/report.py build-hipo/bin/unit_tests check/known-issues
```

On the fork, the workflow `.github/workflows/known-issues.yml` runs it on
each push of `known-issues` (Ubuntu, Release, HiPO) and puts the table in
the job summary: one job on `known-issues` and one per `fixes/*` branch,
which merges that branch into `known-issues` first. The fix branches don't
contain the workflow, so pushing one runs nothing; to report on one of them
after changing it:

```shell
gh workflow run known-issues.yml --ref known-issues -f branch=fixes/<name>
```

Besides this directory, `known-issues` only adds the workflow and a few
lines in `check/CMakeLists.txt`; no fix branch touches either, so the
merges don't conflict. It is also the only workflow enabled on the fork:
upstream's are disabled in the fork's Actions settings.

- On `known-issues`, the job fails on an unexpected pass, a crash or a
  failing regression test.
- With a fix branch merged in, `report.py` finds the fix branches that the
  commit contains (the branch itself and, for a stacked fix, the one
  below). The cases that `fixes.txt` maps to them must pass ("fixed"); one
  that is still present fails the job. Any other case that passes is
  listed as "passes (not in fixes.txt)" without failing it. Crashes and
  failing regression tests fail the job as before.

The report writes issue numbers without a hash and doesn't name the
upstream repository, so that it doesn't link to upstream. Any failed
assertion counts as "still present", including a setup failure, so `-s`
shows which one fired.

To check our fixes together, run it on `latest-with-fixes`: every case in
`fixes.txt` must be "fixed". Run directly, `unit_tests "[known-issue]"`
reports each of them as failed, because each one passes.

## Fixed upstream

These issues of ours are closed. The ones fixed while tracked here keep
their tests as regression tests (`[known-issue-fixed]`); the older ones
have no test here:

| Issue | Fixed by | Test upstream |
|---|---|---|
| #3090 MIP: presolve wrongly declares infeasibility (dominated columns) | #3091 | |
| #3642 wrong "Optimal" with a MIP start whose image presolve cut off | #3762 (with #3634, #3751) | `issue-3642`, `issue-3634` in `check/TestCheckSolution.cpp`; here `issue-3642-mip-start-image-infeasible-after-presolve`. A start objective not attained in the reduced space still cuts off the optimum when the objective is integral: `issue-3642-mip-start-unattained-objective` |
| #3700 highspy writes the log header to stdout at the first `Highs()` | #3697, #3704 (once per process is intended) | |
| #3710 `Highs::crossover` segfaults before any `run()` | #3731 | `issue-3710` in `check/TestCrossover.cpp` |
| #3715 HiPO stops on an offset-free P-D gap (offset in the model) | #3744 | `issue-3715` in `check/TestLpSolvers.cpp`; here `issue-3715-hipo-model-offset`. Offsets created by presolve still fail: `issue-3715-hipo-presolve-offset` |
| #3749 HiPO mishandles the objective sense on maximisation QPs | #3752 (same fix as our `fixes/hipo-maxqp-offdiag`, since dropped) | `issue-3749` in `check/TestHipo.cpp`; here `weakness-hipo-maxqp-offdiag-cycles`, `-offdiag-fixed-column`, `-fixed-column-dual` |
