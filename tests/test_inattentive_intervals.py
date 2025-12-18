from analysis.inattentive_intervals import infer_not_visible_intervals, merge_inattentive_intervals


def test_merge_inattentive_intervals_union_and_kinds():
    raw = [
        {"type": "DROWSY", "start": 0.0, "end": 10.0},
        {"type": "LOOKING_DOWN", "start": 5.0, "end": 8.0},
    ]
    merged = merge_inattentive_intervals(raw, join_gap_sec=0.0)
    assert merged == [{"type": "INATTENTIVE", "start": 0.0, "end": 10.0, "kinds": ["DROWSY", "LOOKING_DOWN"]}]


def test_merge_inattentive_intervals_join_gap():
    raw = [
        {"type": "DROWSY", "start": 0.0, "end": 1.0},
        {"type": "NOT_VISIBLE", "start": 1.2, "end": 2.0},
    ]
    merged = merge_inattentive_intervals(raw, join_gap_sec=0.3)
    assert merged == [{"type": "INATTENTIVE", "start": 0.0, "end": 2.0, "kinds": ["DROWSY", "NOT_VISIBLE"]}]


def test_infer_not_visible_intervals_basic():
    ts = [0.0, 0.5, 1.0, 5.0]
    out = infer_not_visible_intervals(ts, session_end=10.0, gap_sec=1.5, tail_cap_sec=3.0)
    assert out == [
        {"type": "NOT_VISIBLE", "start": 1.0, "end": 5.0},
        {"type": "NOT_VISIBLE", "start": 5.0, "end": 8.0},
    ]

