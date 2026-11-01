"""Cooperative runtime/memory limits checked between resumable work units."""

import resource
import sys
import time


class RunBudgetExceeded(RuntimeError):
    pass


class RunBudget:
    def __init__(self, settings):
        self.settings = settings
        self.started = time.monotonic()

    def check(self):
        elapsed = time.monotonic() - self.started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        gib = peak / (1024**3 if sys.platform == "darwin" else 1024**2)
        if elapsed > self.settings["max_seconds"] or gib > self.settings["max_rss_gib"]:
            raise RunBudgetExceeded(
                f"Pilot/run budget reached: {elapsed:.1f}s, peak {gib:.2f} GiB; "
                "completed stages remain resumable. Use a newly configured run for changed budgets."
            )
