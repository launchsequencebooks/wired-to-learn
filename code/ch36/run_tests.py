# Run the model's test suite, as a CI server would before every release, and print a short report.
import sys
import pytest


class Report:                                              # a tiny pytest plugin that collects the outcomes
    results = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.skipped and hasattr(report, "wasxfail"):   # a known limitation, expected to fail
                outcome, why = "XFAIL", f"({report.wasxfail})"
            elif report.failed:
                crash = getattr(report.longrepr, "reprcrash", None)
                outcome = "FAILED"
                why = (crash.message if crash else str(report.longrepr)).split("\n")[0]
            else:
                outcome, why = report.outcome.upper(), ""
            self.results.append((report.nodeid.split("::")[1], outcome, why))


code = pytest.main(["-p", "no:terminal", "-p", "no:cacheprovider", "-W", "ignore", "test_clothing_model.py"],
                   plugins=[Report()])
for name, outcome, why in Report.results:
    print(f"{outcome:7s} {name} {why}".rstrip())
passed = sum(o in ("PASSED", "XFAIL") for _, o, _ in Report.results)
print(f"{passed} passed or expected to fail, {len(Report.results) - passed} failed -> "
      f"{'release blocked' if code else 'ok to release'}")
sys.exit(code)
