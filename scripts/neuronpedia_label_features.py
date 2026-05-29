"""Exp 2 prong B — rank SAE features by activation at one anchor and pull
Neuronpedia auto-interp labels + top-activating exemplar sentences.

Reads a `sae_feature_firings.py` output `.pt`, restricts to one anchor,
and ranks features two ways:

  - by mean activation across ALL records (zeros for non-firing runs), and
  - by firing frequency (fraction of records where the feature is active).

For the top-N features it queries the Neuronpedia API for the feature's
explanation (label) and a few top-activating text snippets, so we can read
off *what* the Ready representation is made of (class vs attribute vs
task/format features) rather than only whether it is class-discriminative.

Neuronpedia source ids follow `<layer>-<set>` under a model, e.g.
`gemma-3-27b` / `16-gemmascope2-res-65k`. Resolve the exact set with
`--list-sources` (queries the model's available sources) if unsure. A
public read key is usually unnecessary; set NEURONPEDIA_API_KEY to raise
rate limits.

Output: prints two ranked tables and writes a JSON sidecar with the full
feature->label/exemplars mapping.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--firings", required=True,
                   help="Output .pt of sae_feature_firings.py")
    p.add_argument("--anchor", required=True,
                   help="Anchor label to analyze, e.g. end_ready")
    p.add_argument("--top-n", type=int, default=30,
                   help="Number of top features to label per ranking.")
    p.add_argument("--np-model", default="gemma-3-27b",
                   help="Neuronpedia model id.")
    p.add_argument("--np-source", default=None,
                   help="Neuronpedia source id, e.g. '16-gemmascope2-res-65k'. "
                        "Required unless --list-sources or --no-labels.")
    p.add_argument("--list-sources", action="store_true",
                   help="List Neuronpedia sources for --np-model and exit.")
    p.add_argument("--no-labels", action="store_true",
                   help="Skip Neuronpedia; just rank features locally.")
    p.add_argument("--n-exemplars", type=int, default=3,
                   help="Top-activating snippets to keep per feature.")
    p.add_argument("--out", default=None,
                   help="JSON sidecar path. Default: alongside --firings.")
    p.add_argument("--base", default="https://www.neuronpedia.org",
                   help="Neuronpedia base URL.")
    p.add_argument("--sleep", type=float, default=0.2,
                   help="Seconds between API calls (politeness/rate limit).")
    return p.parse_args()


def _http_get(url: str, headers: dict) -> dict | list | None:
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for {url}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"  ERR {type(e).__name__}: {e} for {url}", file=sys.stderr)
    return None


def _headers() -> dict:
    h = {"Accept": "application/json", "User-Agent": "twenty-q-interp/1.0"}
    key = os.environ.get("NEURONPEDIA_API_KEY")
    if key:
        h["X-Api-Key"] = key
    return h


def list_sources(base: str, model: str) -> None:
    # Neuronpedia exposes per-model source sets; try the common endpoints.
    for path in (f"/api/model/{model}", f"/api/{model}/sources"):
        data = _http_get(base + path, _headers())
        if data:
            print(f"--- {path} ---")
            print(json.dumps(data, indent=2)[:4000])
            return
    print("Could not list sources; check --np-model or browse the website.",
          file=sys.stderr)


def fetch_feature(base: str, model: str, source: str, idx: int,
                  n_exemplars: int) -> dict:
    url = f"{base}/api/feature/{model}/{source}/{idx}"
    data = _http_get(url, _headers())
    out: dict = {"index": idx, "label": None, "exemplars": []}
    if not isinstance(data, dict):
        return out
    expls = data.get("explanations") or []
    if expls:
        out["label"] = expls[0].get("description")
    acts = data.get("activations") or []
    for a in acts[:n_exemplars]:
        toks = a.get("tokens") or []
        vals = a.get("values") or []
        if toks and vals:
            j = int(np.argmax(vals))
            lo, hi = max(0, j - 8), min(len(toks), j + 4)
            snippet = "".join(toks[lo:hi]).replace("\n", "\\n")
            out["exemplars"].append(snippet.strip())
        elif a.get("text"):
            out["exemplars"].append(str(a["text"])[:120])
    return out


def main() -> int:
    args = parse_args()
    if args.list_sources:
        list_sources(args.base, args.np_model)
        return 0

    payload = torch.load(args.firings, map_location="cpu", weights_only=False)
    records = [r for r in payload["records"] if r["anchor"] == args.anchor]
    if not records:
        raise SystemExit(f"No records for anchor={args.anchor!r}")
    n = len(records)
    print(f"anchor={args.anchor} n_records={n} "
          f"d_sae={payload['meta'].get('d_sae')}")

    # Aggregate sparse top-k records into per-feature stats.
    sum_act: dict[int, float] = {}
    n_fire: dict[int, int] = {}
    for r in records:
        idx = r["feature_idx"].tolist()
        act = r["activation"].tolist()
        for fid, a in zip(idx, act):
            fid = int(fid)
            sum_act[fid] = sum_act.get(fid, 0.0) + float(a)
            n_fire[fid] = n_fire.get(fid, 0) + 1

    feats = list(sum_act.keys())
    mean_act = {f: sum_act[f] / n for f in feats}          # over ALL records
    fire_freq = {f: n_fire[f] / n for f in feats}
    mean_act_when_fire = {f: sum_act[f] / n_fire[f] for f in feats}

    by_mean = sorted(feats, key=lambda f: mean_act[f], reverse=True)
    by_freq = sorted(feats, key=lambda f: (fire_freq[f], mean_act[f]),
                     reverse=True)

    def _label_block(ranking: list[int]) -> dict[int, dict]:
        labels: dict[int, dict] = {}
        if args.no_labels or not args.np_source:
            return labels
        for fid in ranking[: args.top_n]:
            labels[fid] = fetch_feature(args.base, args.np_model,
                                        args.np_source, fid, args.n_exemplars)
            time.sleep(args.sleep)
        return labels

    union = list(dict.fromkeys(by_mean[: args.top_n] + by_freq[: args.top_n]))
    label_map: dict[int, dict] = {}
    if not args.no_labels and args.np_source:
        print(f"Fetching Neuronpedia labels for {len(union)} features "
              f"({args.np_model}/{args.np_source})...")
        for fid in union:
            label_map[fid] = fetch_feature(args.base, args.np_model,
                                            args.np_source, fid,
                                            args.n_exemplars)
            time.sleep(args.sleep)

    def _print_table(title: str, ranking: list[int]) -> None:
        print(f"\n=== {title} (top {args.top_n}) ===")
        print(f"{'feat':>7} {'mean_act':>9} {'fire%':>6} {'act|fire':>9}  label")
        for fid in ranking[: args.top_n]:
            lab = (label_map.get(fid) or {}).get("label") or ""
            print(f"{fid:>7} {mean_act[fid]:>9.3f} "
                  f"{100*fire_freq[fid]:>5.1f}% {mean_act_when_fire[fid]:>9.3f}"
                  f"  {lab[:70]}")

    _print_table("Features by MEAN ACTIVATION at " + args.anchor, by_mean)
    _print_table("Features by FIRING FREQUENCY at " + args.anchor, by_freq)

    out_path = Path(args.out) if args.out else Path(args.firings).with_suffix(
        f".labels_{args.anchor}.json")
    sidecar = {
        "firings": str(Path(args.firings).resolve()),
        "anchor": args.anchor,
        "n_records": n,
        "np_model": args.np_model,
        "np_source": args.np_source,
        "ranking_by_mean_act": by_mean[: args.top_n],
        "ranking_by_fire_freq": by_freq[: args.top_n],
        "per_feature": {
            str(f): {
                "mean_act": mean_act[f],
                "fire_freq": fire_freq[f],
                "mean_act_when_fire": mean_act_when_fire[f],
                "label": (label_map.get(f) or {}).get("label"),
                "exemplars": (label_map.get(f) or {}).get("exemplars", []),
            }
            for f in union
        },
    }
    out_path.write_text(json.dumps(sidecar, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
