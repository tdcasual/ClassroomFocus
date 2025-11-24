import json
from typing import List, Dict, Any

def plot_timeline(labeled_events: List[Dict[str, Any]], out_png: str, title: str = "ASR Timeline", face_tracks: dict = None, show_text: bool = True, font_path: str = None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import font_manager

    if not labeled_events:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_png, bbox_inches='tight')
        return

    # If a font_path is provided and exists, register it so matplotlib can use it
    font_prop = None
    if font_path:
        try:
            font_manager.fontManager.addfont(font_path)
            import matplotlib as mpl
            font_prop = font_manager.FontProperties(fname=font_path)
            # set rcParams to use this font by family name to avoid path warnings
            try:
                fam = font_prop.get_name()
                mpl.rcParams['font.family'] = fam
            except Exception:
                pass
        except Exception:
            font_prop = None

    t_min = min(e["start"] for e in labeled_events)
    t_max = max(e["end"] for e in labeled_events)
    span = max(1.0, t_max - t_min)
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(t_min - 0.01 * span, t_max + 0.01 * span)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(title)

    colors = {"teacher": "#d62728", "student": "#1f77b4"}

    for i, ev in enumerate(labeled_events):
        start = ev["start"]
        end = ev["end"]
        label = ev.get("label", "student")
        color = colors.get(label, "#7f7f7f")
        rect = Rectangle((start, 0.1), end - start, 0.8, color=color, alpha=0.8)
        ax.add_patch(rect)
        mid = (start + end) / 2.0
        if show_text:
            txt = (ev.get("text") or "").strip()
            if len(txt) > 50:
                txt = txt[:47] + "..."
            if font_prop is not None:
                ax.text(mid, 0.5, txt, ha="center", va="center", fontsize=8, color="white", fontproperties=font_prop)
            else:
                ax.text(mid, 0.5, txt, ha="center", va="center", fontsize=8, color="white")

    # If face_tracks provided, draw teacher presence as a thin bar on top
    if face_tracks:
        # compute teacher id (if any) as the one appearing most often
        teacher_tid = None
        max_count = 0
        for tid, recs in face_tracks.items():
            if len(recs) > max_count:
                max_count = len(recs)
                teacher_tid = tid
        if teacher_tid is not None:
            # draw small occupancy rectangles at y=0.95
            for r in face_tracks.get(teacher_tid, []):
                t = r.get('ts')
                ax.add_patch(Rectangle((t - 0.05, 0.95), 0.1, 0.04, color='black', alpha=0.9))
            if font_prop is not None:
                ax.text(0.99, 0.97, f"teacher:{teacher_tid}", ha='right', va='center', transform=ax.transAxes, fontsize=8, fontproperties=font_prop)
            else:
                ax.text(0.99, 0.97, f"teacher:{teacher_tid}", ha='right', va='center', transform=ax.transAxes, fontsize=8)

    # legend
    ax.plot([], [], color=colors["teacher"], linewidth=6, label="teacher")
    ax.plot([], [], color=colors["student"], linewidth=6, label="student")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')


def save_labeled_jsonl(labeled_events: List[Dict[str, Any]], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in labeled_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outpng", required=True)
    args = parser.parse_args()
    with open(args.infile, "r", encoding="utf-8") as f:
        labeled = [json.loads(l) for l in f if l.strip()]
    plot_timeline(labeled, args.outpng)
