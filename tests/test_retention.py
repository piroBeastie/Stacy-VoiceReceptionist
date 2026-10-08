"""The retention policy, and the sweep that enforces it.

These transcripts are somebody's name, number and address. Two things have to be
true: what is past the window actually goes, and nothing else does — a sweep
that eats the appointment book is worse than one that never runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from stacy.retention import KEEP_FOREVER, RetentionPolicy, sweep

NOW = datetime(2026, 9, 20, 10, 0, tzinfo=ZoneInfo("America/New_York"))


def _daily(path, day: datetime, rows: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"call": f"c{i}"}) for i in range(rows)) + "\n", encoding="utf-8")


def _records(path, stamps: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"message_id": f"m{i}", "created_at": s.isoformat()}) for i, s in enumerate(stamps))
        + "\n",
        encoding="utf-8",
    )


def _tree(root) -> None:
    """A data directory with one file either side of every window."""
    _daily(root / "calls" / "calls-2026-09-19.jsonl", NOW - timedelta(days=1))
    _daily(root / "calls" / "calls-2026-07-01.jsonl", NOW - timedelta(days=81))
    _daily(root / "traces" / "turns-2026-09-19.jsonl", NOW - timedelta(days=1))
    _daily(root / "traces" / "turns-2026-08-01.jsonl", NOW - timedelta(days=50))
    _records(root / "messages" / "messages-millbrook.jsonl", [NOW - timedelta(days=2), NOW - timedelta(days=200)])
    _records(
        root / "calendar" / "appointments-millbrook.jsonl",
        [NOW - timedelta(days=2), NOW - timedelta(days=400)],
    )


class TestWhatGoesAndWhatStays:
    def test_only_what_is_past_its_window_goes(self, tmp_path) -> None:
        _tree(tmp_path)
        done = sweep(tmp_path, now=NOW, apply=True)

        assert (tmp_path / "calls" / "calls-2026-09-19.jsonl").exists()
        assert not (tmp_path / "calls" / "calls-2026-07-01.jsonl").exists()
        assert (tmp_path / "traces" / "turns-2026-09-19.jsonl").exists()
        assert not (tmp_path / "traces" / "turns-2026-08-01.jsonl").exists()
        assert len(done.files_removed) == 2

        kept = (tmp_path / "messages" / "messages-millbrook.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(kept) == 1 and "m0" in kept[0]

    def test_the_appointment_book_is_never_swept_by_default(self, tmp_path) -> None:
        _tree(tmp_path)
        before = (tmp_path / "calendar" / "appointments-millbrook.jsonl").read_text(encoding="utf-8")
        done = sweep(tmp_path, now=NOW, apply=True)
        after = (tmp_path / "calendar" / "appointments-millbrook.jsonl").read_text(encoding="utf-8")
        assert after == before
        assert "appointments" in done.kept_forever

    def test_a_client_policy_can_sweep_the_book_too(self, tmp_path) -> None:
        _tree(tmp_path)
        sweep(tmp_path, RetentionPolicy(appointments_days=365), now=NOW, apply=True)
        kept = (tmp_path / "calendar" / "appointments-millbrook.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(kept) == 1

    def test_without_apply_nothing_is_touched(self, tmp_path) -> None:
        _tree(tmp_path)
        done = sweep(tmp_path, now=NOW)
        assert done.files_removed and done.lines_removed  # it found work
        assert (tmp_path / "calls" / "calls-2026-07-01.jsonl").exists()  # and did none of it
        assert len((tmp_path / "messages" / "messages-millbrook.jsonl").read_text(encoding="utf-8").splitlines()) == 2
        assert "would remove" in done.spoken()

    def test_a_record_with_no_readable_date_is_kept(self, tmp_path) -> None:
        path = tmp_path / "messages" / "messages-millbrook.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"message_id": "m0"}\nnot json at all\n', encoding="utf-8")
        sweep(tmp_path, now=NOW, apply=True)
        # Deleting a customer's record because a field was renamed is the worse failure.
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    def test_an_empty_data_directory_is_not_an_error(self, tmp_path) -> None:
        done = sweep(tmp_path, now=NOW, apply=True)
        assert done.nothing_to_do


class TestThePolicyComesFromTheDeployment:
    def test_windows_are_overridable_and_reported(self) -> None:
        policy, chosen = RetentionPolicy.from_env(
            {"STACY_RETAIN_CALLS_DAYS": "7", "STACY_RETAIN_TRACES_DAYS": "3"}
        )
        assert (policy.calls_days, policy.traces_days) == (7, 3)
        assert policy.messages_days == RetentionPolicy().messages_days  # untouched
        assert chosen == ["STACY_RETAIN_CALLS_DAYS", "STACY_RETAIN_TRACES_DAYS"]

    def test_nothing_set_means_the_defaults_and_says_so(self) -> None:
        policy, chosen = RetentionPolicy.from_env({})
        assert chosen == [] and policy == RetentionPolicy()

    def test_nonsense_falls_back_rather_than_crashing_a_sweep(self) -> None:
        policy, chosen = RetentionPolicy.from_env(
            {"STACY_RETAIN_CALLS_DAYS": "soon", "STACY_RETAIN_TRACES_DAYS": "-5"}
        )
        assert policy == RetentionPolicy() and chosen == []

    def test_zero_means_keep_not_delete_everything(self, tmp_path) -> None:
        _tree(tmp_path)
        done = sweep(tmp_path, RetentionPolicy(calls_days=KEEP_FOREVER), now=NOW, apply=True)
        assert (tmp_path / "calls" / "calls-2026-07-01.jsonl").exists()
        assert "calls" in done.kept_forever
