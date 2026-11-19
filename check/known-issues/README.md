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
each on top of `latest`, or on top of the fix it depends on. The test
headers name them; commit hashes change with every rebase, so nothing refers
to them. The branch `latest-with-fixes` is `latest` with every fix branch
merged.

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
changed the behaviour; read its output. Any failed assertion counts as
"still present", including a setup failure, so `-s` shows which one fired.

To check our fixes, merge `latest-with-fixes` into a scratch branch of
`known-issues` and run again. Now every test case with a fix should be
reported as failed, because each one passes.

## Fixed upstream

These issues of ours are closed. The ones fixed while tracked here keep
their tests as regression tests (`[known-issue-fixed]`); the older ones
have no test here:

| Issue | Fixed by | Test upstream |
|---|---|---|
| #3090 MIP: presolve wrongly declares infeasibility (dominated columns) | #3091 | |
| #3700 highspy writes the log header to stdout at the first `Highs()` | #3697, #3704 (once per process is intended) | |
| #3710 `Highs::crossover` segfaults before any `run()` | #3731 | `issue-3710` in `check/TestCrossover.cpp` |
| #3715 HiPO stops on an offset-free P-D gap (offset in the model) | #3744 | `issue-3715` in `check/TestLpSolvers.cpp`; here `issue-3715-hipo-model-offset`. Offsets created by presolve still fail: `issue-3715-hipo-presolve-offset` |
