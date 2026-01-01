#!/usr/bin/env python3
"""
llama_topk_mass.py — Measure top‑k probability mass over a full generation from llama.cpp server.

Goal
----
For each generated token t, compute the fraction of probability mass captured by the
top‑k candidates (k in a user-provided list, e.g., 100,128,200) compared to the *full*
distribution (which sums to 1.0). The mass for a given k is:
    mass_k(t) = sum_{i in TopK(t)} p_i
where p_i are the *actual* probabilities (exp(logprob)) returned by the server for
the top-N you requested. Because these probabilities are computed from the full softmax
on the server, summing p_i over the returned candidates gives the *exact* captured mass;
the remainder (1 - mass_k) is the mass outside the returned set.

Important server notes
----------------------
• Use the native /completion endpoint with "n_probs = max(k_list)" for large K (100+).
  The OpenAI-compatible /v1/chat/completions endpoint may cap top_logprobs in some builds.
• If you want the mass to reflect the model's *raw* distribution (unclipped by top‑p,
  top‑k, repetition penalties, etc.), set temperature < 0 (e.g., --temperature -1).
  llama.cpp will still compute probabilities via a plain softmax of the raw logits,
  ignoring sampler settings, while sampling greedily. This isolates the underlying model.
• Large K implies larger payloads and slower streaming: expect tens of MB over a few
  hundred tokens if K≈200.

Example
-------
  python llama_topk_mass.py --prompt "Let's reason step by step: prove the binomial theorem." \
      --klist 100,128,200 --max-tokens 400 --endpoint completion \
      --host 127.0.0.1 --port 8080 --temperature -1 --top-p 1.0 --top-k 0 \
      --csv /tmp/mass.csv --json /tmp/mass.json

CSV contains per-step masses; stdout shows a summary.

"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception as e:
    print("This script requires 'requests'. Install with:\n  pip install requests", file=sys.stderr)
    raise

# Types
TokenProb = Tuple[str, float]  # (token string, logprob)
StepMass = Dict[str, float]    # keys include 'mass_k{K}', plus 'chosen_token', etc.


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure top‑k mass over a generation from llama.cpp server.")
    ap.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    ap.add_argument("--endpoint", choices=["completion", "chat", "auto"], default="completion",
                    help="Endpoint to use. For K≥100, 'completion' is recommended. (default: completion)")
    ap.add_argument("--model", default="local", help="Model name for chat endpoint; often ignored by server.")
    ap.add_argument("--prompt", required=True, help="User prompt to send.")
    ap.add_argument("--system", default=None, help="Optional system prompt; only used with chat endpoint.")
    ap.add_argument("--klist", default="100,128,200",
                    help="Comma-separated list of k values to measure (default: 100,128,200).")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate. Default: 512.")
    ap.add_argument("--temperature", type=float, default=-1.0,
                    help="Sampling temperature. Set <0 (e.g., -1) to compute plain-softmax probabilities (recommended). Default: -1.0")
    ap.add_argument("--top-p", type=float, default=1.0, help="top_p for sampling (only used by server if temperature >= 0). Default: 1.0")
    ap.add_argument("--top-k", type=int, default=0, help="top_k for sampling (0 disables). Default: 0")
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="Repetition penalty (1.0 disables). Default: 1.0")
    ap.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout seconds. Default: 600.")
    ap.add_argument("--no-stream", action="store_true", help="Disable streaming; process after completion.")
    ap.add_argument("--post-sampling-probs", action="store_true",
                    help="If set, request post-sampling probabilities (after sampler chain). Default: pre-sampler.")
    ap.add_argument("--csv", default=None, help="Write per-step masses to this CSV path.")
    ap.add_argument("--json", default=None, help="Write per-step masses + summary to this JSON path.")
    ap.add_argument("--verbose", action="store_true", help="Print per-step masses to stdout.")
    return ap.parse_args()


def sse_events(resp: requests.Response) -> Iterator[Dict]:
    """Yield JSON dicts from a text/event-stream ('data: {json}') response."""
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def request_chat(base: str, model: str, prompt: str, system: Optional[str], kmax: int,
                 temperature: float, top_p: float, max_tokens: int, timeout: float, stream: bool
                 ) -> Iterable[Dict]:
    """Yield server responses (streaming events or a single dict) from /v1/chat/completions."""
    url = f"{base}/v1/chat/completions"
    body = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ],
        "stream": bool(stream),
        "logprobs": True,
        "top_logprobs": int(kmax),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    r = requests.post(url, json=body, stream=stream, timeout=timeout)
    r.raise_for_status()
    if stream:
        yield from sse_events(r)
    else:
        yield r.json()


def request_completion(base: str, prompt: str, kmax: int, temperature: float, top_p: float,
                       top_k: int, repeat_penalty: float, max_tokens: int, timeout: float, stream: bool,
                       post_sampling_probs: bool = False
                       ) -> Iterable[Dict]:
    """Yield server responses (streaming events or a single dict) from native /completion."""
    url = f"{base}/completion"
    body = {
        "prompt": prompt,
        "n_predict": int(max_tokens),
        "stream": bool(stream),
        "n_probs": int(kmax),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repeat_penalty": float(repeat_penalty),
        "post_sampling_probs": bool(post_sampling_probs),
    }
    r = requests.post(url, json=body, stream=stream, timeout=timeout)
    r.raise_for_status()
    if stream:
        yield from sse_events(r)
    else:
        yield r.json()


def extract_top_candidates_from_chat_event(evt: Dict) -> Optional[Tuple[str, List[TokenProb]]]:
    """From a chat-completions streaming chunk (or final JSON), extract chosen token and its top candidates."""
    # Handle both streaming and non-streaming (we try streaming first)
    choices = evt.get("choices") or []
    if not choices:
        return None
    ch0 = choices[0]
    # Streaming: token-level info in logprobs.content[0]
    lp = (ch0.get("logprobs") or {}).get("content", [])
    if not lp:
        return None
    item = lp[0]
    chosen_tok = item.get("token")
    top = item.get("top_logprobs") or []
    topk: List[TokenProb] = []
    for cand in top:
        tok = cand.get("token", "")
        logprob = float(cand.get("logprob"))
        topk.append((tok, logprob))
    if chosen_tok is None or not topk:
        return None
    return chosen_tok, topk



def extract_top_candidates_from_completion_event(evt: Dict) -> Optional[Tuple[str, List[TokenProb]]]:
    """From a /completion streaming chunk (or final JSON), extract chosen token and top candidates.

    Supports both default (top_logprobs + logprob) and post-sampling (top_probs + prob).
    """
    probs = evt.get("completion_probabilities")
    if not probs:
        return None
    last = probs[-1]
    chosen_tok = last.get("token") or ""
    # When post_sampling_proba=true, fields are 'top_probs' and 'prob'; otherwise 'top_logprobs' and 'logprob'
    top = last.get("top_logprobs")
    post_mode = False
    if top is None:
        top = last.get("top_probs") or []
        post_mode = True
    topk: List[TokenProb] = []
    for cand in top:
        tok = cand.get("token", "")
        if post_mode:
            p = float(cand.get("prob"))
            # store as logprob consistently
            logprob = math.log(max(p, 1e-45))
        else:
            logprob = float(cand.get("logprob"))
        topk.append((tok, logprob))
    if not chosen_tok or not topk:
        return None
    return chosen_tok, topk


def compute_masses_for_step(topk: List[TokenProb], k_values: List[int]) -> Dict[int, float]:
    """Return {K: mass_topK} using the actual probabilities (exp(logprob))."""
    # Sort by logprob descending to be safe
    topk_sorted = sorted(topk, key=lambda x: x[1], reverse=True)
    probs = [math.exp(lp) for _, lp in topk_sorted]
    # prefix sums
    prefix = []
    s = 0.0
    for p in probs:
        s += p
        prefix.append(s)
    masses: Dict[int, float] = {}
    for K in k_values:
        if K <= 0:
            masses[K] = 0.0
        elif K <= len(prefix):
            masses[K] = prefix[K-1]
        else:
            masses[K] = prefix[-1]  # can't exceed what server returned
    return masses


def summarize(name: str, values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    vs = sorted(values)
    n = len(vs)
    def pct(p: float) -> float:
        if n == 1:
            return vs[0]
        idx = max(0, min(n-1, int(round(p*(n-1)))))
        return vs[idx]
    return {
        "count": float(n),
        "mean": float(stats.fmean(vs)),
        "median": float(stats.median(vs)),
        "min": float(vs[0]),
        "max": float(vs[-1]),
        "p05": float(pct(0.05)),
        "p95": float(pct(0.95)),
        "pct_ge_0_90": float(100.0*sum(v>=0.90 for v in vs)/n),
        "pct_ge_0_95": float(100.0*sum(v>=0.95 for v in vs)/n),
        "pct_ge_0_99": float(100.0*sum(v>=0.99 for v in vs)/n),
    }


def main() -> None:
    args = parse_args()
    k_values = sorted({int(k.strip()) for k in args.klist.split(",") if k.strip()})
    if not k_values:
        print("Provide at least one k via --klist", file=sys.stderr)
        sys.exit(2)
    kmax = max(k_values)

    base = f"http://{args.host}:{args.port}"
    stream = not args.no_stream

    # Storage for per-step results
    per_steps: List[Dict[str, float]] = []
    k_mass_lists: Dict[int, List[float]] = {K: [] for K in k_values}
    step_idx = 0
    using_chat = False
    using_completion = False

    def handle(chosen: str, topk: List[TokenProb]) -> None:
        nonlocal step_idx
        masses = compute_masses_for_step(topk, k_values)
        row: Dict[str, float] = {"step": float(step_idx)}
        row["len_top_returned"] = float(len(topk))
        # store masses
        for K, m in masses.items():
            row[f"mass_k{K}"] = m
            k_mass_lists[K].append(m)
        # for reference, store the probability of the chosen token if present
        chosen_lp = None
        for tok, lp in topk:
            if tok == chosen:
                chosen_lp = lp
                break
        if chosen_lp is not None:
            row["p_chosen_in_top"] = math.exp(chosen_lp)
        else:
            row["p_chosen_in_top"] = float("nan")
        per_steps.append(row)
        if args.verbose:
            m_str = " ".join([f"k={K}: {masses[K]*100:5.2f}%" for K in k_values])
            print(f"[t={step_idx:03d}] {m_str}")
        step_idx += 1

    # Make requests
    try:
        if args.endpoint in ("auto", "chat"):
            using_chat = True
            for evt in request_chat(base, args.model, args.prompt, args.system, kmax,
                                    args.temperature, args.top_p,
                                    args.max_tokens, args.timeout, stream):
                got = extract_top_candidates_from_chat_event(evt)
                if got:
                    chosen, topk = got
                    handle(chosen, topk)
            # If we requested a large k but got fewer, advise using /completion
            if step_idx > 0 and any(row["len_top_returned"] < kmax for row in per_steps):
                print(f"Warning: chat endpoint returned fewer than requested top_logprobs for some steps. "
                      f"Consider --endpoint completion for large K.", file=sys.stderr)
                if args.endpoint == "chat":
                    # If user forced chat, we exit with what we have
                    pass
                elif args.endpoint == "auto" and step_idx == 0:
                    # fall back if nothing came
                    using_chat = False
        if (args.endpoint in ("auto", "completion")) and (not using_chat or step_idx == 0):
            using_completion = True
            for evt in request_completion(base, args.prompt, kmax, args.temperature, args.top_p,
                                          args.top_k, args.repeat_penalty,
                                          args.max_tokens, args.timeout, stream,
                                          post_sampling_probs=args.post_sampling_probs):
                got = extract_top_candidates_from_completion_event(evt)
                if got:
                    chosen, topk = got
                    handle(chosen, topk)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(3)
    except requests.RequestException as e:
        print(f"Request error: {e}", file=sys.stderr)
        sys.exit(4)

    if step_idx == 0:
        print("No token‑level probability data received. Ensure your server supports logprobs/n_probs.", file=sys.stderr)
        sys.exit(5)

    # Summaries
    print("\n=== Top‑k mass summary over generated steps ===")
    for K in k_values:
        vals = k_mass_lists[K]
        S = summarize(f"k={K}", vals)
        print(f"k={K:>4} | steps={int(S['count']):>4} | mean={S['mean']*100:6.2f}% | median={S['median']*100:6.2f}% "
              f"| min={S['min']*100:6.2f}% | p05={S['p05']*100:6.2f}% | p95={S['p95']*100:6.2f}% "
              f"| ≥90%:{S['pct_ge_0_90']:5.1f}% | ≥95%:{S['pct_ge_0_95']:5.1f}% | ≥99%:{S['pct_ge_0_99']:5.1f}%")

    # Optional outputs
    if args.csv:
        try:
            import csv  # stdlib
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["step", "len_top_returned"] + [f"mass_k{K}" for K in k_values] + ["p_chosen_in_top"]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in per_steps:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
            print(f"Wrote CSV to {args.csv}")
        except Exception as e:
            print(f"Failed to write CSV: {e}", file=sys.stderr)

    if args.json:
        try:
            summary = {f"k{K}": summarize(f"k={K}", k_mass_lists[K]) for K in k_values}
            out = {"k_values": k_values, "endpoint_used": "chat" if using_chat and not using_completion else "completion",
                   "per_step": per_steps, "summary": summary}
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"Wrote JSON to {args.json}")
        except Exception as e:
            print(f"Failed to write JSON: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
