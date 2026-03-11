import argparse
import json
import threading
import time
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import requests

API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}


@dataclass
class WorkerStats:
    requests_sent: int = 0
    errors: int = 0


class ThroughputMeter:
    def __init__(self, test_seconds: int):
        self.start_time = time.time()
        self.test_seconds = test_seconds
        self.tokens_per_second = [0 for _ in range(test_seconds)]
        self.lock = threading.Lock()
        # thread lock to ensure thread-safe updates to tokens_per_second

    def add_token(self, now_ts: float) -> None:
        sec = int(now_ts - self.start_time)
        if 0 <= sec < self.test_seconds:
            with self.lock:
                self.tokens_per_second[sec] += 1


def build_payload(model: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }  # parameters injection


def worker_loop(worker_id: int, meter: ThroughputMeter, stop_event: threading.Event, payload: dict, timeout_s: int, stats: WorkerStats) -> None:
    session = requests.Session()
    # measurement loop, same as eval_llm.py
    while not stop_event.is_set():
        stats.requests_sent += 1
        try:
            with session.post(API_URL, headers=HEADERS, json=payload, stream=True, timeout=timeout_s) as resp:
                if resp.status_code != 200:
                    stats.errors += 1
                    continue

                for chunk in resp.iter_lines():
                    if stop_event.is_set():
                        break
                    if not chunk:
                        continue

                    line = chunk.decode("utf-8", errors="ignore")
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        meter.add_token(time.time())

        except requests.RequestException:
            stats.errors += 1


def run_ramp_test(test_seconds: int, timeout_s: int, payload: dict) -> tuple[List[int], List[int], List[WorkerStats]]:
    meter = ThroughputMeter(test_seconds=test_seconds)
    stop_event = threading.Event()
    # ramp test: start with 1 worker, add 1 worker every second until test_seconds, 
    # each worker sends requests in a loop until stop_event is set
    workers: List[threading.Thread] = []
    worker_stats: List[WorkerStats] = []
    active_workers_per_second: List[int] = []

    print(f"Start ramp test for {test_seconds} seconds")
    print("Rule: +1 worker per second")

    for sec in range(test_seconds):
        stats = WorkerStats()
        worker_stats.append(stats)

        t = threading.Thread(
            target=worker_loop,
            args=(sec + 1, meter, stop_event, payload, timeout_s, stats),
            daemon=True,
        )
        workers.append(t)
        t.start()

        active_workers_per_second.append(len(workers))
        print(f"second={sec + 1:02d}, active_workers={len(workers)}")
        time.sleep(1)

    stop_event.set()

    for t in workers:
        t.join(timeout=3)

    return active_workers_per_second, meter.tokens_per_second, worker_stats

# throughput plotting script
def plot_curve(concurrency: List[int], throughput: List[int], output_path: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(concurrency, throughput, marker="o", linewidth=2)
    plt.title("LLM Throughput Curve (Ramp: +1 req/s)")
    plt.xlabel("Concurrency Level (active workers)")
    plt.ylabel("Throughput (tokens/second)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM throughput under increasing concurrency")
    parser.add_argument("--seconds", type=int, default=20, help="Total test duration in seconds")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per request in seconds")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generated tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument(
        "--model",
        type=str,
        default="/root/.cache/huggingface/Qwen3-4B-quantized.w4a16",
        help="Model name/path served by vLLM",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Briefly introduce Jetson Orin NX and list key features.",
        help="Prompt text",
    )
    parser.add_argument("--output", type=str, default="llm_throughput.png", help="Output figure path")
    args = parser.parse_args()

    payload = build_payload(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    concurrency, throughput, stats = run_ramp_test(
        test_seconds=args.seconds,
        timeout_s=args.timeout,
        payload=payload,
    )

    plot_curve(concurrency, throughput, args.output)

    total_requests = sum(s.requests_sent for s in stats)
    total_errors = sum(s.errors for s in stats)

    print("\n===== Summary =====")
    print(f"concurrency points: {len(concurrency)}")
    print(f"total requests sent: {total_requests}")
    print(f"total request errors: {total_errors}")
    print(f"peak throughput: {max(throughput) if throughput else 0} tokens/s")
    print(f"saved figure: {args.output}")


if __name__ == "__main__":
    main()
