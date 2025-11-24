import tempfile
import json
from analysis.teacher_labeler import load_asr_jsonl, load_face_tracks_jsonl, choose_teacher_track, label_asr_segments


def make_asr(tmpdir, events):
    p = tmpdir / "asr.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return str(p)


def make_faces(tmpdir, recs):
    p = tmpdir / "faces.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for ev in recs:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return str(p)


def test_choose_teacher_and_label(tmp_path):
    # create two face tracks: track A centered and big, track B small at side
    faces = [
        {"ts": 0.5, "track_id": "A", "bbox": [0.2, 0.2, 0.4, 0.4]},
        {"ts": 1.5, "track_id": "A", "bbox": [0.21, 0.2, 0.39, 0.4]},
        {"ts": 0.6, "track_id": "B", "bbox": [0.8, 0.2, 0.05, 0.06]},
    ]
    asr = [
        {"start": 0.4, "end": 0.7, "text": "hello"},
        {"start": 1.3, "end": 1.6, "text": "question"},
        {"start": 2.0, "end": 2.5, "text": "student talk"},
    ]
    fpath = make_faces(tmp_path, faces)
    apath = make_asr(tmp_path, asr)

    tracks = load_face_tracks_jsonl(fpath)
    assert isinstance(tracks, dict)
    teacher = choose_teacher_track(tracks)
    assert teacher == "A"

    events = load_asr_jsonl(apath)
    labeled = label_asr_segments(events, tracks)
    # first two segments overlap with teacher A
    assert labeled[0]["label"] == "teacher"
    assert labeled[1]["label"] == "teacher"
    assert labeled[2]["label"] == "student"
