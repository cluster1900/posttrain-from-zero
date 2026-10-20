from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SGT = timezone(timedelta(hours=8))


def test_training_interval_timezone_and_elapsed():
    chapter = (ROOT / 'chapters/07_sft.md').read_text()
    cases = [
        ('2026-10-12T12:27:58.921+00:00', '2026-10-12T20:12:58.140+00:00',
         '10-12 20:27:58.921', '10-13 04:12:58.140', 27899219),
        ('2026-10-12T20:16:48.015+00:00', '2026-10-13T08:02:03.136+00:00',
         '10-13 04:16:48.015', '10-13 16:02:03.136', 42315121),
    ]
    for start, end, sgt_start, sgt_end, milliseconds in cases:
        a, b = datetime.fromisoformat(start), datetime.fromisoformat(end)
        assert (b - a) // timedelta(milliseconds=1) == milliseconds
        assert a.astimezone(SGT).strftime('%m-%d %H:%M:%S.%f')[:-3] == sgt_start
        assert b.astimezone(SGT).strftime('%m-%d %H:%M:%S.%f')[:-3] == sgt_end
        assert sgt_start in chapter
        assert sgt_end.split(' ', 1)[1] in chapter
