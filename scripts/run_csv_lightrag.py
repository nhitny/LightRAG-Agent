#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error


def is_section_header(cell0: str, cell1: str) -> bool:
    c0 = (cell0 or "").strip()
    c1 = (cell1 or "").strip()
    if c1:
        return False
    if len(c0) >= 2 and c0[1] == "." and c0[0].isalpha():
        return True
    return False


def pick_question(row: list[str], col_index: int | None) -> str:
    if col_index is not None:
        if col_index < len(row):
            return (row[col_index] or "").strip()
        return ""
    if len(row) > 1 and (row[1] or "").strip():
        return (row[1] or "").strip()
    return (row[0] or "").strip() if row else ""


def login(base_url: str, username: str, password: str) -> str:
    url = f"{base_url.rstrip('/')}/login"
    data = urllib.parse.urlencode({"username": username, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Login failed {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Login connection error: {e}")

    obj = json.loads(body)
    token = obj.get("access_token")
    if not token:
        raise RuntimeError(f"Login response missing access_token: {body}")
    return token


def call_query(base_url: str, token: str, query: str, mode: str, include_references: bool) -> str:
    url = f"{base_url.rstrip('/')}/query"
    payload = {
        "query": query,
        "mode": mode,
        "include_references": include_references,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Query error {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Query connection error: {e}")

    obj = json.loads(body)
    return (obj.get("response") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Call LightRAG /query for questions in a CSV and write answers.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--base-url", default="http://localhost:9622", help="LightRAG base URL")
    parser.add_argument("--token", default=None, help="Bearer token (optional)")
    parser.add_argument("--username", default=None, help="Login username (if token not provided)")
    parser.add_argument("--password", default=None, help="Login password (if token not provided)")
    parser.add_argument("--mode", default="mix", help="Query mode: local/global/hybrid/naive/mix/bypass")
    parser.add_argument("--include-references", action="store_true", help="Include references in response (unused in CSV)")
    parser.add_argument("--question-col", type=int, default=None, help="0-based index of question column")
    parser.add_argument("--answer-col-name", default="Answer", help="Column name to store answers")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests")
    args = parser.parse_args()

    token = args.token or os.getenv("LIGHTRAG_TOKEN")
    if not token:
        if not args.username or not args.password:
            raise SystemExit("Provide --token or set LIGHTRAG_TOKEN, or pass --username/--password")
        token = login(args.base_url, args.username, args.password)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise SystemExit("Empty CSV")

    header = rows[0]
    if args.answer_col_name in header:
        answer_col_index = header.index(args.answer_col_name)
    else:
        answer_col_index = len(header)
        header.append(args.answer_col_name)

    out_rows = [header]
    for i, row in enumerate(rows[1:], start=2):
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))

        q = pick_question(row, args.question_col)
        if not q:
            out_rows.append(row)
            continue

        if is_section_header(row[0] if len(row) > 0 else "", row[1] if len(row) > 1 else ""):
            out_rows.append(row)
            continue

        try:
            answer = call_query(args.base_url, token, q, args.mode, args.include_references)
        except Exception as e:
            answer = f"[ERROR] {e}"

        row[answer_col_index] = answer
        out_rows.append(row)
        time.sleep(args.sleep)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
