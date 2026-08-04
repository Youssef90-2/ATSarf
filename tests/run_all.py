"""
run_all.py — run every regression test in this folder.

    py -3.11 tests/run_all.py

None of these need CAMeL: token flags come from lexicons.py + the phrase
matcher, or are synthesized by hand. That is deliberate — the FSM and graph
logic must be verifiable without the morphology layer installed.

check_graph.py is a HARNESS, not a test: it rebuilds the narrator graph from
a <book>.clean.hadiths.json and reports the pass conditions. Run it directly
with a book argument:

    py -3.11 tests/check_graph.py kafi1.clean.hadiths.json 0.5
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TESTS = sorted(p for p in HERE.glob("test_*.py"))


def main():
    env_note = "utf-8 output enforced per test file"
    print(f"running {len(TESTS)} test files ({env_note})\n")
    failed = []
    for path in TESTS:
        result = subprocess.run([sys.executable, str(path)],
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        tail = [ln for ln in (result.stdout or "").strip().splitlines()
                if "passed" in ln]
        summary = tail[-1] if tail else "(no summary)"
        status = "OK  " if result.returncode == 0 else "FAIL"
        print(f"  {status}  {path.name:<16} {summary}")
        if result.returncode != 0:
            failed.append((path.name, result.stdout, result.stderr))

    if failed:
        print("\n" + "=" * 60)
        for name, out, err in failed:
            print(f"--- {name} ---")
            print(out)
            if err.strip():
                print(err)
    print()
    print("ALL GREEN" if not failed else f"{len(failed)} FILE(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
