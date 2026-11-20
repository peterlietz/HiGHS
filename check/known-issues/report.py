"""Report which known issues are still present and which pass unexpectedly.

usage: python3 -I report.py UNIT_TESTS KNOWN_ISSUES_DIR [SUMMARY_FILE]

Runs every [known-issue] and [known-issue-fixed] test case of UNIT_TESTS on
its own and prints a Markdown table; with SUMMARY_FILE (for example
$GITHUB_STEP_SUMMARY) it appends the table there too.

On known-issues (latest plus the reproducers), the exit status is 1 if a
known issue passes unexpectedly (upstream fixed it), a test case crashes,
or a regression test of a fixed issue fails.

With fix branches merged in (CI merges each one into known-issues), the
cases that fixes.txt maps to a fix branch HEAD contains must pass, and the exit
status is 1 if one of them is still present. Another case that passes is
reported, but is no failure: fixes.txt may lack it, or it may need several
fixes together. A fix branch counts as contained if it, or origin/<it>, is
an ancestor of HEAD, so CI needs the full history (fetch-depth: 0).
Crashes and failing regression tests fail the job in either mode.

Issue numbers are written as "issue N", never with a hash, and the upstream
repository is not named, so that nothing in the report links to upstream.
"""

import glob
import os
import re
import subprocess
import sys

TIMEOUT = 1800


def list_cases(unit_tests, tag):
    out = subprocess.run([unit_tests, tag, "--list-test-names-only"],
                         capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines()
            if l.strip() and not l.strip().endswith("test cases")]


def neutral(text):
    text = text.replace("ERGO-Code/HiGHS", "")
    return re.sub(r"#(\d+)", r"issue \1", text).strip()


def headlines(source_dir):
    """Maps each test case name to its file and the file's header summary."""
    result = {}
    for path in sorted(glob.glob(os.path.join(source_dir, "Test*.cpp"))):
        lines = open(path).read().splitlines()
        header = []
        for line in lines:
            if not line.startswith("//"):
                break
            text = line[2:].strip()
            if not text:
                break
            header.append(text)
        # The origin is the first line, continued while a parenthesis is open
        n = 1
        while n < len(header) and \
                " ".join(header[:n]).count("(") > " ".join(header[:n]).count(")"):
            n += 1
        origin = neutral(" ".join(header[:n]))
        summary = neutral(" ".join(header[n:]))
        source = open(path).read()
        for name in re.findall(r'TEST_CASE\(\s*"([^"]+)"', source):
            result[name] = (os.path.basename(path), origin, summary)
    return result


def read_fixes(source_dir):
    """Maps each test case name to the fix branches listed in fixes.txt."""
    path = os.path.join(source_dir, "fixes.txt")
    mapping = {}
    if not os.path.exists(path):
        return mapping
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, _, branches = line.partition(":")
        mapping[name.strip()] = branches.split()
    return mapping


def contained_fixes(source_dir, branches):
    """The fix branches whose tip (local or origin/) is an ancestor of HEAD."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=source_dir,
                              capture_output=True, text=True)
    contained = set()
    for branch in sorted(branches):
        for ref in (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}"):
            if git("rev-parse", "--verify", "--quiet", ref).returncode != 0:
                continue
            if git("merge-base", "--is-ancestor", ref, "HEAD").returncode == 0:
                contained.add(branch)
                break
    return contained


def run_case(unit_tests, name):
    try:
        proc = subprocess.run([unit_tests, name], capture_output=True,
                              text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    output = proc.stdout + proc.stderr
    if proc.returncode < 0 or proc.returncode >= 128 or \
            "FATAL ERROR" in output:
        return "crash", output
    return ("pass" if proc.returncode == 0 else "fail"), output


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    unit_tests, source_dir = sys.argv[1], sys.argv[2]
    summary_file = sys.argv[3] if len(sys.argv) > 3 else None
    info = headlines(source_dir)
    fixes = read_fixes(source_dir)
    contained = contained_fixes(source_dir,
                                {b for bs in fixes.values() for b in bs})
    names = set(list_cases(unit_tests, "[known-issue]"))
    for name in sorted(set(fixes) - names):
        print(f"fixes.txt names no [known-issue] case: {name}",
              file=sys.stderr)
    rows = []
    counts = {}
    bad = False
    for tag, kind in (("[known-issue]", "known"), ("[known-issue-fixed]",
                                                    "fixed")):
        for name in list_cases(unit_tests, tag):
            result, output = run_case(unit_tests, name)
            if kind == "known":
                # A [!shouldfail] case exits 0 while the problem is present
                fixed_by = [b for b in fixes.get(name, []) if b in contained]
                if result not in ("pass", "fail"):
                    status = "**" + result + "**"
                elif fixed_by:
                    status = {"pass": "**still present, fix expected**",
                              "fail": "fixed"}[result]
                elif contained:
                    status = {"pass": "still present",
                              "fail": "passes (not in fixes.txt)"}[result]
                else:
                    status = {"pass": "still present",
                              "fail": "**unexpected pass**"}[result]
            else:
                status = {"pass": "regression test passes",
                          "fail": "**regression test fails**"}.get(
                              result, "**" + result + "**")
            if status.startswith("**"):
                bad = True
                print(f"---- {name}: {status}\n{output}", file=sys.stderr)
            counts[status] = counts.get(status, 0) + 1
            source, origin, summary = info.get(name, ("?", "", ""))
            rows.append((name, status, source, origin, summary))
            print(f"{status:28s} {name}", flush=True)

    lines = ["## Known issues", ""]
    if contained:
        lines += ["Fix branches contained: " +
                  ", ".join(f"`{b}`" for b in sorted(contained)), ""]
    lines.append(", ".join(f"{n} {s.strip('*')}" for s, n in
                           sorted(counts.items())))
    lines += ["", "| Test case | Status | File | Origin | Problem |",
              "|---|---|---|---|---|"]
    for name, status, source, origin, summary in rows:
        cells = [f"`{name}`", status, f"`{source}`", origin, summary]
        lines.append("| " + " | ".join(c.replace("|", "\\|")
                                         for c in cells) + " |")
    if contained:
        lines += ["", "\"fixed\": a case that fixes.txt maps to a contained "
                  "fix branch passes. \"still present, fix expected\": "
                  "such a case still fails, so that fix doesn't work here. "
                  "\"passes (not in fixes.txt)\": an unmapped case passes; "
                  "add it to fixes.txt if one branch fixes it on its own."]
    else:
        lines += ["", "An unexpected pass means that latest changed: check "
                  "the upstream change and keep the test as a regression "
                  "test (see check/known-issues/README.md)."]
    report = "\n".join(lines) + "\n"
    print(report)
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(report)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
