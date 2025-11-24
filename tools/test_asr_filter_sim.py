"""Simulate ASR events to demonstrate final-first + stability-fallback filtering.

This reproduces the buffering logic added to demo_video_session.py and
prints accepted ASR segments.
"""
import time

def simulate(events, asr_stability_secs=0.8, asr_bucket_width=0.5):
    asr_buffers = {}
    def _is_final_raw(raw):
        if not isinstance(raw, dict):
            return False
        for k in ("is_final", "final", "is_last", "status"):
            v = raw.get(k)
            if v and (v is True or (isinstance(v, str) and v in ("final", "completed"))):
                return True
        if raw.get("status") in ("final", "completed"):
            return True
        return False
    def _bucket_for_ts(ts):
        return int(ts // asr_bucket_width)
    accepted = []
    def _accept_buffer(bucket):
        buf = asr_buffers.get(bucket)
        if not buf:
            return
        text = buf.get("final_text") or buf.get("last_text") or ""
        ev = {"ts": float(buf.get("first_ts", 0.0)), "text": text, "raw": buf.get("raw_last")}
        accepted.append(ev)
        del asr_buffers[bucket]
    # simulate
    for ev in events:
        ts = ev.get('ts', 0.0)
        text = ev.get('text','')
        raw = ev.get('raw',{})
        now = ev.get('now', ts)
        bucket = _bucket_for_ts(ts)
        buf = asr_buffers.get(bucket)
        if buf is None:
            buf = {
                'first_ts': ts,
                'last_text': text,
                'raw_last': raw,
                'last_change_time': now,
                'last_seen_time': now,
                'final_text': None,
            }
            asr_buffers[bucket] = buf
        else:
            if text != buf.get('last_text'):
                buf['last_text'] = text
                buf['raw_last'] = raw
                buf['last_change_time'] = now
            buf['last_seen_time'] = now
        if _is_final_raw(raw):
            buf['final_text'] = text
            _accept_buffer(bucket)
            continue
        if now - buf.get('last_change_time', now) >= asr_stability_secs:
            _accept_buffer(bucket)
            continue
    # flush remaining
    for b in list(asr_buffers.keys()):
        _accept_buffer(b)
    return accepted

if __name__ == '__main__':
    # Construct a timeline of events simulating progressive partials
    events = []
    # Segment at ts=2.0 with three progressive hypotheses
    events.append({'ts':2.0, 'text':'h', 'raw':{}, 'now':2.0})
    events.append({'ts':2.0, 'text':'he', 'raw':{}, 'now':2.2})
    events.append({'ts':2.0, 'text':'hello', 'raw':{}, 'now':3.1})
    # Segment at ts=5.0 with partials then a final
    events.append({'ts':5.0, 'text':'t', 'raw':{}, 'now':5.0})
    events.append({'ts':5.0, 'text':'te', 'raw':{}, 'now':5.2})
    events.append({'ts':5.0, 'text':'test', 'raw':{'is_final':True}, 'now':5.3})
    # Segment at ts=8.0 with quick stable partials
    events.append({'ts':8.0, 'text':'a', 'raw':{}, 'now':8.0})
    events.append({'ts':8.0, 'text':'a', 'raw':{}, 'now':8.9})

    accepted = simulate(events, asr_stability_secs=0.8, asr_bucket_width=0.5)
    print('Accepted segments:')
    for a in accepted:
        print(a)
