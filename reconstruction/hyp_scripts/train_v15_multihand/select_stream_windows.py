#!/usr/bin/env python3
"""Select one stream from a JSONL window manifest."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    rows = []
    with Path(args.input).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("stream_id")) == args.stream_id:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No windows for {args.stream_id} in {args.input}")
    rows.sort(key=lambda row: (int(row["start"]), int(row["end"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({
        "stream_id": args.stream_id,
        "windows": len(rows),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()

