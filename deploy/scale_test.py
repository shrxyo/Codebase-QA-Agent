"""Scale and failure test for the docker compose stack (stdlib only).

Start the stack with the debug endpoint on (and a short visibility timeout
if you run the worker-crash check):

    DEBUG_MODE=true CELERY_VISIBILITY_TIMEOUT=60 docker compose up -d --build
    python deploy/scale_test.py                       # all checks except worker crash
    python deploy/scale_test.py --only throughput     # just the load test
    python deploy/scale_test.py --only worker_crash   # ~2 min

Checks:
  round_robin     /health through the LB reaches every API replica
  dedup           5 concurrent forced /index calls -> exactly one build
  any_replica     a repo indexed by the worker is served by every replica
  throughput      retrieval requests/s, p50/p95 at fixed concurrency
  replica_down    kill an API replica under load: how many requests fail
  worker_crash    kill the worker mid-build: the job is redelivered and finishes

Throughput uses /debug/retrieval (no LLM): the hosted model's rate limit
(8k tokens/min) would make /chat measure the provider, not this system.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PROJECT = "codebase-qa-agent"
QUESTIONS = [
    "How does it check internet connectivity?",
    "What does the timeout option do?",
    "Which file is the entry point?",
    "How are fallback URLs used?",
    "Where is the browser implementation?",
]


def call(base: str, method: str, path: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:  # e.g. nginx's HTML 502 page
            return e.code, {}


def containers(service: str) -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "--filter", f"label=com.docker.compose.project={PROJECT}",
         "--filter", f"label=com.docker.compose.service={service}", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return sorted(out)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


def wait_indexed(base: str, repo: str, timeout: float = 600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, d = call(base, "GET", "/index/status?repo_url=" + urllib.parse.quote(repo))
        if code == 200 and d["status"] in ("indexed", "failed"):
            return d
        time.sleep(1)
    raise TimeoutError(f"{repo} not indexed after {timeout}s")


def ensure_indexed(base: str, repo: str) -> dict:
    code, d = call(base, "POST", "/index", {"repo_url": repo})
    assert code in (200, 202), (code, d)
    return d if d["status"] == "indexed" else wait_indexed(base, repo)


# ---------------------------------------------------------------------------

def check_round_robin(base, repo, args):
    seen = Counter(call(base, "GET", "/health")[1]["replica"] for _ in range(30))
    n = len(containers("api"))
    print(f"  30 /health calls hit {len(seen)} of {n} replicas: {dict(seen)}")
    return len(seen) == n


def check_dedup(base, repo, args):
    ensure_indexed(base, repo)
    with ThreadPoolExecutor(5) as pool:
        res = list(pool.map(lambda _: call(base, "POST", "/index", {"repo_url": repo, "force": True}), range(5)))
    ids = {d["build_id"] for _, d in res}
    replicas = {d["replica"] for _, d in res}
    codes = [c for c, _ in res]
    print(f"  5 concurrent forced /index -> codes {codes}, {len(ids)} build id(s), via {len(replicas)} replica(s)")
    print(f"  chat stayed available during the rebuild: ready={[d['ready'] for _, d in res]}")
    t = time.perf_counter()
    final = wait_indexed(base, repo)
    print(f"  rebuild finished in {time.perf_counter() - t:.1f}s, build {final['build_id'][:8]}, "
          f"timings_ms={final['timings_ms']}")
    return len(ids) == 1 and final["build_id"] in ids and all(d["ready"] for _, d in res)


def check_any_replica(base, repo, args):
    ensure_indexed(base, repo)
    seen, bad = Counter(), 0
    for i in range(30):
        code, d = call(base, "POST", "/debug/retrieval", {"repo_url": repo, "question": QUESTIONS[i % 5]})
        if code != 200 or not d["chunks"]:
            bad += 1
            continue
        seen[d["replica"]] += 1
    n = len(containers("api"))
    print(f"  30 retrievals: {30 - bad} ok, served by {len(seen)} of {n} replicas: {dict(seen)}")
    return bad == 0 and len(seen) == n


def _load(base, repo, concurrency, seconds, on_tick=None):
    """Closed-loop load: `concurrency` clients, each sending back to back."""
    lat, errors, replicas = [], Counter(), Counter()
    lock = threading.Lock()
    stop = time.time() + seconds

    def client(i):
        k = i
        while time.time() < stop:
            t = time.perf_counter()
            try:
                code, d = call(base, "POST", "/debug/retrieval",
                               {"repo_url": repo, "question": QUESTIONS[k % 5]}, timeout=30)
            except Exception as exc:  # connection reset etc.
                code, d = type(exc).__name__, {}
            ms = (time.perf_counter() - t) * 1000
            with lock:
                if code == 200:
                    lat.append(ms)
                    replicas[d["replica"]] += 1
                else:
                    errors[str(code)] += 1
            k += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency + 1) as pool:
        futs = [pool.submit(client, i) for i in range(concurrency)]
        if on_tick:
            pool.submit(on_tick)
        for f in futs:
            f.result()
    wall = time.perf_counter() - t0
    return lat, errors, replicas, wall


def check_throughput(base, repo, args):
    ensure_indexed(base, repo)
    _load(base, repo, args.concurrency, 3)  # warm the per-replica indexer caches
    lat, errors, replicas, wall = _load(base, repo, args.concurrency, args.seconds)
    n = len(containers("api"))
    print(f"  api replicas={n} concurrency={args.concurrency} for {args.seconds}s: "
          f"{len(lat) / wall:.1f} req/s, p50 {pct(lat, 50):.0f} ms, p95 {pct(lat, 95):.0f} ms, "
          f"errors {sum(errors.values())} {dict(errors)}")
    print(f"  per replica: {dict(replicas)}")
    print("  RESULT " + json.dumps({"replicas": n, "concurrency": args.concurrency, "seconds": args.seconds,
                                    "requests": len(lat), "rps": round(len(lat) / wall, 2),
                                    "p50_ms": round(pct(lat, 50)), "p95_ms": round(pct(lat, 95)),
                                    "errors": sum(errors.values())}))
    return not errors


def check_replica_down(base, repo, args):
    ensure_indexed(base, repo)
    victim = containers("api")[-1]
    events = []

    def chaos():
        time.sleep(5)
        subprocess.run(["docker", "kill", victim], capture_output=True, check=True)
        events.append(f"t=5s killed {victim}")
        time.sleep(10)
        subprocess.run(["docker", "start", victim], capture_output=True, check=True)
        events.append(f"t=15s restarted {victim}")

    lat, errors, replicas, wall = _load(base, repo, 4, 30, on_tick=chaos)
    total = len(lat) + sum(errors.values())
    for e in events:
        print("  " + e)
    print(f"  {total} requests in {wall:.0f}s, {sum(errors.values())} failed {dict(errors)}, "
          f"p95 {pct(lat, 95):.0f} ms")
    print(f"  per replica: {dict(replicas)}")
    return sum(errors.values()) <= 4  # at most the in-flight requests on the victim


def check_worker_crash(base, repo, args):
    big = args.crash_repo
    ensure_indexed(base, repo)
    code, d = call(base, "POST", "/index", {"repo_url": big, "force": True})
    build = d["build_id"]
    t0 = time.perf_counter()
    while True:
        code, d = call(base, "GET", "/index/status?repo_url=" + urllib.parse.quote(big))
        if d["status"] == "indexing":
            break
        time.sleep(0.5)
    time.sleep(5)  # well into cloning/embedding
    worker = containers("worker")[0]
    subprocess.run(["docker", "kill", worker], capture_output=True, check=True)
    print(f"  t={time.perf_counter() - t0:.0f}s killed {worker} mid-build (status=indexing)")
    subprocess.run(["docker", "start", worker], capture_output=True, check=True)
    print(f"  t={time.perf_counter() - t0:.0f}s restarted {worker}; waiting for Redis to redeliver the job")
    final = wait_indexed(base, big, timeout=900)
    print(f"  t={time.perf_counter() - t0:.0f}s status={final['status']} build {final['build_id'][:8]} "
          f"(same build: {final['build_id'] == build}), chunks={final['chunks_created']}")
    return final["status"] == "indexed" and final["build_id"] == build


CHECKS = {
    "round_robin": check_round_robin,
    "dedup": check_dedup,
    "any_replica": check_any_replica,
    "throughput": check_throughput,
    "replica_down": check_replica_down,
    "worker_crash": check_worker_crash,
}
DEFAULT = ["round_robin", "dedup", "any_replica", "throughput", "replica_down"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--repo", default="https://github.com/sindresorhus/is-online")
    ap.add_argument("--crash-repo", default="https://github.com/psf/requests")
    ap.add_argument("--only", choices=CHECKS, action="append")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=20)
    args = ap.parse_args()

    code, ready = call(args.base, "GET", "/ready")
    if code != 200:
        sys.exit(f"stack not ready: {ready}")
    results = {}
    for name in args.only or DEFAULT:
        print(f"[{name}]")
        results[name] = CHECKS[name](args.base, args.repo, args)
        print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
