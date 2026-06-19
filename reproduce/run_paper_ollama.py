import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc


# -----------------------------
# Ollama helpers (requests)
# -----------------------------
def ollama_post(base_url: str, path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    """Call Ollama HTTP API and return parsed JSON."""
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ollama_get(base_url: str, path: str, timeout: int = 300) -> dict[str, Any]:
    """Call Ollama HTTP GET API and return parsed JSON."""
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ensure_model_exists(base_url: str, model_name: str) -> None:
    """Validate that a model exists in local Ollama registry."""
    tags = ollama_get(base_url, "/api/tags", timeout=30)
    available = {m.get("name", "") for m in tags.get("models", [])}
    if model_name not in available:
        raise RuntimeError(
            f"Model '{model_name}' chưa có trong Ollama. Hãy chạy: ollama pull {model_name}"
        )


def detect_embedding_dim(base_url: str, embedding_model: str) -> int:
    """Detect embedding dimension from Ollama /api/embed."""
    data = ollama_post(
        base_url,
        "/api/embed",
        {"model": embedding_model, "input": ["dimension probe"]},
        timeout=60,
    )
    emb = data["embeddings"][0]
    return len(emb)


# -----------------------------
# LightRAG custom model funcs
# -----------------------------
async def ollama_llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> str:
    """
    Custom llm_model_func for LightRAG using Ollama /api/chat.

    LightRAG will pass extra kwargs such as:
    - hashing_kv
    - keyword_extraction
    - _priority
    - stream
    """
    history_messages = history_messages or []

    base_url = kwargs.pop("base_url")
    model = kwargs.pop("model")
    request_timeout = int(kwargs.pop("request_timeout", 300))

    # Ignore LightRAG internal kwargs not needed by Ollama request body.
    kwargs.pop("hashing_kv", None)
    kwargs.pop("_priority", None)
    kwargs.pop("max_tokens", None)

    keyword_extraction = kwargs.pop("keyword_extraction", False)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    # LightRAG keyword extraction expects JSON-like output.
    if keyword_extraction:
        payload["format"] = "json"

    if "options" in kwargs and isinstance(kwargs["options"], dict):
        payload["options"].update(kwargs.pop("options"))

    data = await asyncio.to_thread(
        ollama_post, base_url, "/api/chat", payload, request_timeout
    )
    return data["message"]["content"]


async def ollama_embedding_func(
    texts: list[str],
    base_url: str,
    embedding_model: str,
    max_token_size: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Custom embedding func for LightRAG using Ollama /api/embed."""
    _ = max_token_size
    request_timeout = int(kwargs.pop("request_timeout", 300))

    payload: dict[str, Any] = {
        "model": embedding_model,
        "input": texts,
    }

    if "options" in kwargs and isinstance(kwargs["options"], dict):
        payload["options"] = kwargs["options"]

    data = await asyncio.to_thread(
        ollama_post, base_url, "/api/embed", payload, request_timeout
    )
    return np.array(data["embeddings"], dtype=np.float32)


# -----------------------------
# Reproduce steps
# -----------------------------
def extract_unique_contexts(input_dir: Path, output_dir: Path) -> list[Path]:
    """Step-0: extract unique `context` fields from each *.jsonl file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"Không tìm thấy file *.jsonl trong: {input_dir}")

    for file_path in jsonl_files:
        name = file_path.stem
        out_path = output_dir / f"{name}_unique_contexts.json"
        unique: dict[str, None] = {}

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                context = obj.get("context")
                if isinstance(context, str) and context:
                    unique[context] = None

        contexts = list(unique.keys())
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(contexts, f, ensure_ascii=False, indent=2)

        print(f"[Step-0] {file_path.name}: {len(contexts)} unique contexts -> {out_path}")
        outputs.append(out_path)

    return outputs


def build_question_generation_prompt(context_summaries: str) -> str:
    """Prompt equivalent to reproduce/Step_2.py but for local LLM."""
    return f"""
Given the following description of a dataset:

{context_summaries}

Please identify 5 potential users who would engage with this dataset.
For each user, list 5 tasks they would perform with this dataset.
Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

Output exactly in this structure:
- User 1: [user description]
    - Task 1: [task description]
        - Question 1: ...
        - Question 2: ...
        - Question 3: ...
        - Question 4: ...
        - Question 5: ...
""".strip()


def compact_context(text: str, max_chars: int = 2400) -> str:
    """Create a compact context snippet to keep prompt size manageable."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head} ... {tail}"


async def generate_questions_file(
    *,
    base_url: str,
    llm_model: str,
    unique_contexts_file: Path,
    questions_file: Path,
) -> None:
    """Step-2 (local): generate question set with qwen via Ollama."""
    with unique_contexts_file.open("r", encoding="utf-8") as f:
        contexts = json.load(f)

    # Use at most first 120 contexts to control prompt size/time.
    snippets = [compact_context(c) for c in contexts[:120]]
    prompt = build_question_generation_prompt("\n\n".join(snippets))

    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You generate structured benchmark questions for RAG evaluation.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    data = await asyncio.to_thread(ollama_post, base_url, "/api/chat", payload, 600)
    text = data["message"]["content"]

    questions_file.parent.mkdir(parents=True, exist_ok=True)
    with questions_file.open("w", encoding="utf-8") as f:
        f.write(text)

    print(f"[Step-2] Generated questions -> {questions_file}")


def extract_queries(questions_file: Path) -> list[str]:
    """Extract '- Question n: ...' from generated question file."""
    data = questions_file.read_text(encoding="utf-8").replace("**", "")
    queries = re.findall(r"- Question \d+: (.+)", data)
    return [q.strip() for q in queries if q.strip()]


async def initialize_rag(
    *,
    working_dir: Path,
    base_url: str,
    llm_model: str,
    embedding_model: str,
    embedding_dim: int,
    request_timeout: int,
    worker_timeout: int,
    llm_max_async: int,
    embedding_max_async: int,
) -> LightRAG:
    """Create and initialize a LightRAG instance with custom Ollama funcs."""
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_max_async=llm_max_async,
        embedding_func_max_async=embedding_max_async,
        default_llm_timeout=worker_timeout,
        llm_model_name=llm_model,
        llm_model_func=lambda prompt, system_prompt=None, history_messages=None, **kwargs: ollama_llm_model_func(  # type: ignore
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=base_url,
            model=llm_model,
            request_timeout=request_timeout,
            **kwargs,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            model_name=embedding_model,
            func=lambda texts, **kwargs: ollama_embedding_func(  # type: ignore
                texts,
                base_url=base_url,
                embedding_model=embedding_model,
                request_timeout=request_timeout,
                **kwargs,
            ),
        ),
    )
    await rag.initialize_storages()
    return rag


async def insert_contexts(rag: LightRAG, unique_contexts_file: Path) -> None:
    """Step-1: insert unique contexts into LightRAG."""
    with unique_contexts_file.open("r", encoding="utf-8") as f:
        contexts = json.load(f)

    retries = 3
    for attempt in range(1, retries + 1):
        try:
            await rag.ainsert(contexts)
            print(f"[Step-1] Inserted {len(contexts)} contexts from {unique_contexts_file}")
            return
        except Exception as e:
            if attempt == retries:
                raise
            print(f"[Step-1] Insert fail ({attempt}/{retries}): {e}; retrying...")
            await asyncio.sleep(5)


async def run_queries(
    rag: LightRAG,
    queries: list[str],
    mode: str,
    result_file: Path,
    error_file: Path,
) -> None:
    """Step-3: run benchmark queries and save outputs."""
    result_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    param = QueryParam(mode=mode)

    for idx, q in enumerate(queries, start=1):
        try:
            ans = await rag.aquery(q, param=param)
            results.append({"query": q, "result": ans})
            print(f"[Step-3] ({idx}/{len(queries)}) done")
        except Exception as e:
            errors.append({"query": q, "error": str(e)})
            print(f"[Step-3] ({idx}/{len(queries)}) error: {e}")

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with error_file.open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"[Step-3] Results -> {result_file}")
    print(f"[Step-3] Errors  -> {error_file}")


async def run_for_class(
    *,
    cls_name: str,
    unique_contexts_file: Path,
    args: argparse.Namespace,
) -> None:
    """Run Step-1/2/3 for one dataset class file."""
    print(f"\n=== Running class: {cls_name} ===")

    workdir = Path(args.work_root) / cls_name
    questions_file = Path(args.out_root) / "questions" / f"{cls_name}_questions.txt"
    result_file = Path(args.out_root) / "results" / f"{cls_name}_{args.mode}_result.json"
    error_file = Path(args.out_root) / "results" / f"{cls_name}_{args.mode}_errors.json"

    rag = await initialize_rag(
        working_dir=workdir,
        base_url=args.ollama_base_url,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        request_timeout=args.request_timeout,
        worker_timeout=args.worker_timeout,
        llm_max_async=args.llm_max_async,
        embedding_max_async=args.embedding_max_async,
    )

    try:
        await insert_contexts(rag, unique_contexts_file)

        if args.questions_dir:
            # Use pre-existing question file from user.
            src = Path(args.questions_dir) / f"{cls_name}_questions.txt"
            if not src.exists():
                raise FileNotFoundError(f"Không tìm thấy file câu hỏi: {src}")
            questions_file.parent.mkdir(parents=True, exist_ok=True)
            questions_file.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[Step-2] Using existing questions: {src}")
        else:
            await generate_questions_file(
                base_url=args.ollama_base_url,
                llm_model=args.llm_model,
                unique_contexts_file=unique_contexts_file,
                questions_file=questions_file,
            )

        queries = extract_queries(questions_file)
        if not queries:
            raise RuntimeError(
                f"Không tách được query từ file {questions_file}. Kiểm tra định dạng '- Question n: ...'"
            )

        await run_queries(rag, queries, args.mode, result_file, error_file)
    finally:
        await rag.llm_response_cache.index_done_callback()
        await rag.finalize_storages()


async def main_async(args: argparse.Namespace) -> None:
    ensure_model_exists(args.ollama_base_url, args.llm_model)
    ensure_model_exists(args.ollama_base_url, args.embedding_model)

    if args.embedding_dim is None:
        args.embedding_dim = detect_embedding_dim(args.ollama_base_url, args.embedding_model)
        print(f"Detected embedding_dim={args.embedding_dim} for {args.embedding_model}")

    unique_dir = Path(args.out_root) / "unique_contexts"
    extracted_files = extract_unique_contexts(Path(args.input_dir), unique_dir)

    for p in extracted_files:
        cls_name = p.name.replace("_unique_contexts.json", "")
        await run_for_class(cls_name=cls_name, unique_contexts_file=p, args=args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce LightRAG pipeline with local Ollama (qwen2.5:7b-instruct)."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing dataset *.jsonl files")
    parser.add_argument("--out-root", default="./reproduce_output", help="Output root for generated files")
    parser.add_argument("--work-root", default="./reproduce_workdir", help="LightRAG working directories root")
    parser.add_argument("--questions-dir", default="", help="Optional existing questions directory")

    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-dim", type=int, default=None)

    parser.add_argument("--mode", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--request-timeout", type=int, default=1200, help="HTTP timeout (seconds) for each Ollama request")
    parser.add_argument("--worker-timeout", type=int, default=1800, help="LightRAG LLM worker timeout (seconds)")
    parser.add_argument("--llm-max-async", type=int, default=1, help="Max concurrent LLM calls in LightRAG")
    parser.add_argument("--embedding-max-async", type=int, default=2, help="Max concurrent embedding calls in LightRAG")

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main_async(cli_args))
    request_timeout = int(kwargs.pop("request_timeout", 300))
