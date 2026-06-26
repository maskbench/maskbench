"""Throttled progress logging for the persistent run log.

tqdm already renders a live progress bar on stderr. Writing `progress.__str__()`
to the file log on *every* item (as the first progress-logging pass did) appends
~one bar-string per video — tens of thousands of near-duplicate lines per run,
plus per-iteration I/O. These helpers log only at milestones (default ~every 5%)
and on the final item. Pure stdlib so it is unit-testable without the heavy deps.
"""
import logging


def should_log_progress(done: int, total: int, every_pct: int = 5) -> bool:
    """True at roughly every ``every_pct`` percent and on the final item."""
    if total <= 0:
        return True
    step = max(1, total * every_pct // 100)
    return done % step == 0 or done >= total


def log_progress(done: int, total: int, label: str, every_pct: int = 5) -> None:
    """Emit a milestone progress line to the file log (throttled)."""
    if should_log_progress(done, total, every_pct):
        pct = (done * 100 // total) if total else 100
        logging.info(f"{label}: {done}/{total} ({pct}%)")
