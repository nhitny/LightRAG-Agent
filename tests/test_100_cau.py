import requests
import csv
import os
import time

# ===== CONFIG =====
API_URL = "http://localhost:9622/query"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMSIsImV4cCI6MTc3NTM3ODIzMCwicm9sZSI6InVzZXIiLCJtZXRhZGF0YSI6eyJhdXRoX21vZGUiOiJlbmFibGVkIn19.jJ4I4lOP5hJl-UkV5h416BGpysH2QduD5xM_gjMyfoM"

INPUT_FILE = "/workspace/nhitny/LightRAG/tests/lightrag_100_questions.csv"
OUTPUT_FILE = "/workspace/nhitny/LightRAG/tests/lightrag_100_questions_out.csv"
OUTPUT_FIX_FILE = "/workspace/nhitny/LightRAG/tests/lightrag_100_questions_out_fix.csv"

MAX_PRINT_CHARS = 200

START_INDEX = 1
END_INDEX = 100

BATCH_SIZE = 10
SLEEP_SECONDS = 0  # không nghỉ

# Nếu True: đọc OUTPUT_FILE, gửi lại những câu bị lỗi/thiếu output,
# rồi ghi toàn bộ kết quả (cả câu có đáp án và chưa có) sang OUTPUT_FIX_FILE.
RETRY_FROM_OUTPUT = True
# ==================


def query_lightrag(question):
    try:
        response = requests.post(
            API_URL,
            json={"query": question},
            headers={"Authorization": f"Bearer {TOKEN}"},
            timeout=60
        )
        data = response.json()
        if "response" in data:
            return data["response"]
        if "answer" in data:
            return data["answer"]
        return str(data)
    except Exception as e:
        return f"ERROR: {e}"


def short_text(text, limit=200):
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

def is_error_output(text):
    if text is None:
        return True
    t = text.strip()
    if not t:
        return True
    if t.startswith("ERROR:"):
        return True
    return False

def load_output_map(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out = {}
    for row in rows:
        stt_raw = (row.get("stt") or "").strip()
        if not stt_raw.isdigit():
            continue
        out[int(stt_raw)] = row.get("output", "")
    return out


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "output" not in fieldnames:
        fieldnames.append("output")

    file_exists = os.path.exists(OUTPUT_FILE)
    processed = 0
    output_map = load_output_map(OUTPUT_FILE) if RETRY_FROM_OUTPUT else {}
    fix_file_exists = os.path.exists(OUTPUT_FIX_FILE)

    for row in rows:
        stt_raw = (row.get("stt") or "").strip()
        if not stt_raw.isdigit():
            continue
        stt = int(stt_raw)

        if stt < START_INDEX or stt > END_INDEX:
            continue

        query = (row.get("query") or "").strip()
        if not query:
            continue

        prev_output = output_map.get(stt, "")
        need_retry = RETRY_FROM_OUTPUT and is_error_output(prev_output)

        if RETRY_FROM_OUTPUT:
            if need_retry:
                print(f"Q{stt}: {query}")
                answer = query_lightrag(query)
                print(f"A{stt}: {short_text(answer, MAX_PRINT_CHARS)}\n")
                row["output"] = answer
            else:
                row["output"] = prev_output

            with open(OUTPUT_FIX_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not fix_file_exists:
                    writer.writeheader()
                    fix_file_exists = True
                writer.writerow(row)
        else:
            print(f"Q{stt}: {query}")
            answer = query_lightrag(query)
            print(f"A{stt}: {short_text(answer, MAX_PRINT_CHARS)}\n")

            row["output"] = answer

            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerow(row)

        processed += 1
        if SLEEP_SECONDS > 0 and processed % BATCH_SIZE == 0:
            print(f"Đã chạy {processed} câu. Nghỉ {SLEEP_SECONDS} giây...\n")
            time.sleep(SLEEP_SECONDS)

    if RETRY_FROM_OUTPUT:
        print("\nSaved to:", OUTPUT_FIX_FILE)
    else:
        print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
