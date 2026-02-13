"""Load teaching moments and print timestamped YouTube links for verification."""

import json
import sys
from pathlib import Path


def load_moments(data_dir: Path):
    moments_dir = data_dir / "teaching_moments"
    moments = []
    for f in sorted(moments_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                moments.append(json.loads(line))
    return moments


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    moments = load_moments(data_dir)

    print(f"Found {len(moments)} moments across {len({m['video_id'] for m in moments})} videos\n")

    for vid_id in dict.fromkeys(m["video_id"] for m in moments):
        vid_moments = [m for m in moments if m["video_id"] == vid_id]
        title = vid_moments[0].get("video_title", vid_id)
        print(f"=== {title} ===")
        print(f"    {len(vid_moments)} moments | model: {vid_moments[0].get('extraction_model', '?')}\n")

        for m in sorted(vid_moments, key=lambda x: x["stop_timestamp"]):
            ts = int(m["stop_timestamp"])
            print(f"  Stop {m['stop_order']}/{m['total_stops']}  [{fmt_time(m['stop_timestamp'])}]")
            print(f"    https://youtu.be/{vid_id}?t={ts}")
            print(f"    dimension: {m['musical_dimension']}  severity: {m['severity']}  type: {m['feedback_type']}")
            print(f"    summary: {m['feedback_summary']}")
            print()


if __name__ == "__main__":
    main()
