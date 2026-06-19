#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
import urllib.request
import urllib.error

API_URL = "https://api.openai.com/v1/responses"


def is_section_header(cell0: str, cell1: str) -> bool:
    c0 = (cell0 or "").strip()
    c1 = (cell1 or "").strip()
    if c1:
        return False
    # Heuristic: headers like "A." or "B." or "A. Something"
    if len(c0) >= 2 and c0[1] == "." and c0[0].isalpha():
        return True
    return False


def pick_question(row: list[str], col_index: int | None) -> str:
    if col_index is not None:
        if col_index < len(row):
            return (row[col_index] or "").strip()
        return ""
    # Auto-detect: prefer column 1 if present, else column 0
    if len(row) > 1 and (row[1] or "").strip():
        return (row[1] or "").strip()
    return (row[0] or "").strip() if row else ""


def call_openai(api_key: str, model: str, prompt: str, max_output_tokens: int) -> str:
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_output_tokens": max_output_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"OpenAI API error {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI API connection error: {e}")

    obj = json.loads(body)
    # Extract text output from Responses API
    output_text = ""
    for item in obj.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    output_text += part.get("text", "")
    return output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Call OpenAI API for questions in a CSV and write answers.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--question-col", type=int, default=None, help="0-based index of question column")
    parser.add_argument("--answer-col-name", default="Answer", help="Column name to store answers")
    parser.add_argument("--max-output-tokens", type=int, default=512, help="Max output tokens per answer")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

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
        # Ensure row length
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
            answer = call_openai(api_key, args.model, q, args.max_output_tokens)
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
