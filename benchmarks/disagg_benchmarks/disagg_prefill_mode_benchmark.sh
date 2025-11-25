#!/bin/bash

# Benchmark disaggregated prefill throughput for:
# 1. Baseline: both prefill and decode run via `vllm serve` (no prefill-only opts).
# 2. Optimized: prefill node launched via `vllm prefill` (hybrid prefill enabled).

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  # vLLM now names the process with VLLM prefix after https://github.com/vllm-project/vllm/pull/21445
  pgrep VLLM | xargs -r kill -9
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

launch_disagg_prefill_baseline() {
  CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
    --port 8100 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"SharedStorageConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --port 8200 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"SharedStorageConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_prefill_proxy_server.py &
  sleep 1
}

launch_disagg_prefill_prefillmode() {
  CUDA_VISIBLE_DEVICES=0 vllm prefill $MODEL_NAME \
    --port 8100 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"SharedStorageConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --port 8200 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"SharedStorageConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_prefill_proxy_server.py &
  sleep 1
}

benchmark() {
  results_folder="./results"
  dataset_name="sonnet"
  dataset_path="../sonnet_8x.txt"
  num_prompts=100
  qps=$1
  prefix_len=50
  input_len=$MAX_MODEL_LEN
  output_len=$2
  tag=$3

  vllm bench serve \
    --backend vllm \
    --model $MODEL_NAME \
    --dataset-name $dataset_name \
    --dataset-path $dataset_path \
    --sonnet-input-len $input_len \
    --sonnet-output-len "$output_len" \
    --sonnet-prefix-len $prefix_len \
    --num-prompts $num_prompts \
    --port 8000 \
    --save-result \
    --result-dir $results_folder \
    --result-filename "$tag"-qps-"$qps".json \
    --request-rate "$qps"

  sleep 2
}

main() {
  MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-1.5B-Instruct"}
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  cd ..
  # Create a long sonnet file so that we can sample long contexts.
  echo "" > sonnet_8x.txt
  for _ in {1..8}; do
    cat sonnet.txt >> sonnet_8x.txt
  done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_output_len=6
  export VLLM_HOST_IP=$(hostname -i | awk '{print $1}')

  launch_disagg_prefill_prefillmode
  for qps in 2 4 6 8; do
    benchmark $qps $default_output_len prefill_mode
  done
  kill_gpu_processes

  launch_disagg_prefill_baseline
  for qps in 2 4 6 8; do
    benchmark $qps $default_output_len serve_prefill
  done
  kill_gpu_processes
}

main "$@"
