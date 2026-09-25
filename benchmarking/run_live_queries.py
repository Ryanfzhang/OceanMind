#!/usr/bin/env python3
"""Ask one shard of live questions through the Next.js query streaming route."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def ask(opener, url: str, question: str, timeout: float) -> dict:
    request = Request(url, data=json.dumps({"query": question}, ensure_ascii=False).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    event_types: Counter[str] = Counter()
    final = None
    stream_errors = []
    started = time.monotonic()
    try:
        with opener.open(request, timeout=timeout) as response:
            http_status = response.status
            content_type = response.headers.get("Content-Type", "")
            for raw in response:
                try:
                    event = json.loads(raw)
                except (UnicodeDecodeError, ValueError) as exc:
                    stream_errors.append(f"invalid_ndjson:{type(exc).__name__}")
                    continue
                kind = event.get("event") if isinstance(event, dict) else None
                if not isinstance(kind, str):
                    stream_errors.append("missing_event_type")
                    continue
                event_types[kind] += 1
                if kind == "final":
                    final = event.get("payload")
                elif kind == "error":
                    stream_errors.append(str((event.get("payload") or {}).get("detail", "stream error"))[:500])
    except HTTPError as exc:
        http_status, content_type = exc.code, exc.headers.get("Content-Type", "")
        stream_errors.append(exc.read(2000).decode("utf-8", errors="replace"))
    except (OSError, URLError) as exc:
        http_status, content_type = None, ""
        stream_errors.append(f"{type(exc).__name__}: {exc}")
    return {"elapsed_seconds": round(time.monotonic() - started, 2),
            "http_status": http_status, "content_type": content_type,
            "event_types": dict(event_types), "stream_errors": stream_errors,
            "final": final}


def issues(case: dict, result: dict) -> list[str]:
    found = []
    final = result["final"]
    if result["http_status"] != 200:
        found.append("http_error")
    if "application/x-ndjson" not in result["content_type"]:
        found.append("frontend_stream_content_type")
    if result["stream_errors"]:
        found.append("stream_error")
    if not isinstance(final, dict):
        return [*found, "missing_final"]
    if final.get("status") != "completed":
        found.append("query_not_completed")
    synthesis = final.get("synthesis") or {}
    if final.get("status") == "completed" and not str(synthesis.get("summary") or "").strip():
        found.append("empty_answer")
    if case["category"] in {"base", "interpretation", "water_mass_mechanism", "novel_code"}:
        if final.get("routing_mode") != "dataset_analysis":
            found.append("data_query_without_analysis")
    if final.get("routing_mode") == "dataset_analysis":
        if not final.get("step_cards"):
            found.append("analysis_without_visible_steps")
        if not final.get("result_cards"):
            found.append("analysis_without_visible_results")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3001/api/query/stream")
    parser.add_argument("--queries", type=Path, default=Path(__file__).with_name("live_queries.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("shard must be between 0 and shards-1")
    cases = json.loads(args.queries.read_text(encoding="utf-8"))
    selected = [case for case in cases if (case["case_id"] - 1) % args.shards == args.shard]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["case_id"])
            except (KeyError, ValueError):
                pass
    opener = build_opener(ProxyHandler({}))
    with args.output.open("a", encoding="utf-8") as out:
        for case in selected:
            if case["case_id"] in done:
                continue
            result = ask(opener, args.url, case["query"], args.timeout)
            record = {"case_id": case["case_id"], "source_query_id": case["source_query_id"],
                      "category": case["category"], "query": case["query"], **result}
            record["issues"] = issues(case, result)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"Q{case['case_id']:03d} {result['http_status']} "
                  f"{(result['final'] or {}).get('status')} {result['elapsed_seconds']}s "
                  f"{','.join(record['issues']) or 'ok'}", flush=True)


if __name__ == "__main__":
    main()
