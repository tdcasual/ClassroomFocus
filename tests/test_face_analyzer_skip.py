from cv.face_analyzer import refresh_results_timestamps


def test_refresh_results_timestamps_updates_dicts():
    results = [{"ts": 1.0, "track_id": 0}, {"ts": 2.0, "track_id": 1}, "skip"]
    refresh_results_timestamps(results, 9.5)
    assert results[0]["ts"] == 9.5
    assert results[1]["ts"] == 9.5
