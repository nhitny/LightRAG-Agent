import requests
import json
import csv

# ===== CONFIG =====

API_URL = "http://localhost:9622/query"

INPUT_FILE = "lightrag_retrieval_test_17.json"

OUTPUT_FILE = "lightrag_answers.csv"

NUM_TEST = 5   # số câu muốn test (None = test tất cả)

# ==================


def query_lightrag(question):

    try:
        response = requests.post(
            API_URL,
            json={"query": question},
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


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        tests = json.load(f)

    # giới hạn số câu test
    if NUM_TEST is not None:
        tests = tests[:NUM_TEST]

    results = []

    for i, item in enumerate(tests, start=1):

        query = item["query"]

        print(f"Test {i}: {query}")

        answer = query_lightrag(query)

        results.append({
            "query": query,
            "lightrag_answer": answer
        })

    # ghi CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["query", "lightrag_answer"]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()