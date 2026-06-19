cd /workspace/nhitny/LightRAG && \
CUDA_VISIBLE_DEVICES=7 taskset -c 0-15 python reproduce/run_paper_ollama.py \
  --input-dir /dataset/UltraDomain_paper4 \
  --out-root ./reproduce_output \
  --work-root ./reproduce_workdir \
  --ollama-base-url http://127.0.0.1:11434 \
  --llm-model qwen2.5:7b-instruct \
  --embedding-model nomic-embed-text:latest \
  --mode hybrid \
  --request-timeout 1800 \
  --worker-timeout 1800 \
  --llm-max-async 1 \
  --embedding-max-async 2

