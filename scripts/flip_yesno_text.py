"""M4 phase 2d (c-text) — Yes/no answer text-level flip behavioral test.

For each kept run T in a self-chosen collection, regenerate the reveal
under counterfactual dialogues where exactly one yes/no answer text has
been flipped (Yes -> No or No -> Yes). Pure behavior — no patching,
just rebuilding the chat context with one answer literal changed.

Two outputs per trial:
1. Greedy reveal text + parsed canonical class (the behavioral signal
   that established improvisation in D-40).
2. **Logit-lens reading** at the pre_reveal_gen position, across ALL
   layers, for ALL 20 bank-class first-tokens. This addresses the
   "suppressed pre-commitment" worry: even if the original class isn't
   the argmax at the final layer, it might still be elevated mid-network
   and get suppressed by late layers (a documented pattern, e.g. negative
   name movers in IOI). The per-layer logit trajectory for the original
   class vs the flipped-consistent class lets us distinguish:
   - Pure improvisation: orig_class logit tracks any-other-class baseline
     across all layers.
   - Suppressed pre-commitment: orig_class logit is elevated mid-network
     then decays toward final.
   - Concurrent consideration: orig_class rises alongside new class until
     new overtakes.

Caveat on early layers: logit lens at L0-~L15 is generally noisy because
the residual stream isn't yet in the unembedding's basis. Focus
interpretation on L20-L48.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twenty_q.banks import load_bank
from twenty_q.config import MODEL_MAIN
from twenty_q.dialogue import (
    REVEAL_USER_MESSAGE,
    ModelHandle,
    _build_chat_input_ids,
    _history_to_chat_turns,
    load_model,
    parse_reveal_to_canonical,
)
from twenty_q.manifest import RunManifest
from twenty_q.permutations import Permutation
from twenty_q.prompts import self_chosen_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Diagnostic run directory.")
    p.add_argument("--out-json", required=True, help="Output path for trial records.")
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--prompt-variant", default="default")
    p.add_argument("--n-per-class", type=int, default=20,
                   help="Subsample to first K runs of each realized class. Default 20 "
                        "(matches M3 scale-up methodology).")
    p.add_argument("--turns-to-flip", type=str, default="1,2,3,4",
                   help="Comma-separated turn indices (1..4) to flip. Default all 4.")
    p.add_argument("--logit-lens", action="store_true",
                   help="Capture per-layer logit-lens readings at pre_reveal_gen "
                        "across all 20 bank-class first-tokens. Adds ~2x to walltime.")
    return p.parse_args()


def _load_kept_manifests(run_dir: Path) -> list[RunManifest]:
    out: list[RunManifest] = []
    for a in sorted(run_dir.glob("attempt_*")):
        mp = a / "manifest.json"
        if not mp.exists():
            continue
        m = RunManifest.load(mp)
        if m.reveal_canonical_id is not None:
            out.append(m)
    return out


def _flip_turn_n_answer(manifest: RunManifest, n: int) -> tuple[list, str, str]:
    """Return (modified_turns, original_answer_text, flipped_answer_text)."""
    turns = [copy.copy(t) for t in manifest.turns]
    orig_raw = turns[n - 1].raw_model_output.strip()
    orig_lower = orig_raw.lower()
    if orig_lower.startswith("yes"):
        flipped_raw = "No"
    elif orig_lower.startswith("no"):
        flipped_raw = "Yes"
    else:
        return turns, orig_raw, ""  # unparseable; caller should skip
    turns[n - 1].raw_model_output = flipped_raw
    turns[n - 1].answer_bool = not turns[n - 1].answer_bool if turns[n - 1].answer_bool is not None else None
    return turns, orig_raw, flipped_raw


def _build_full_reveal_inputs(
    handle: ModelHandle,
    manifest: RunManifest,
    turns_override: list,
    bank,
    prompt_variant: str,
) -> dict[str, torch.Tensor]:
    """Build the full chat ending at the reveal-generation prompt, using a
    (possibly modified) turn list for the turn answers."""
    display_names = {c.id: c.display for c in bank.candidates}
    perm = Permutation(order=tuple(manifest.permutation))
    rendered = self_chosen_prompt(perm, display_names, variant=prompt_variant)
    extra = [
        *_history_to_chat_turns(manifest.ready_raw_output, turns_override),
        {"role": "user", "content": REVEAL_USER_MESSAGE},
    ]
    return _build_chat_input_ids(handle, rendered, extra_turns=extra)


def _bank_first_token_ids(handle: ModelHandle, bank) -> dict[str, int]:
    """Map every bank candidate id -> first token id of " <Display>"
    (Gemma SentencePiece picks up the leading space)."""
    out: dict[str, int] = {}
    for c in bank.candidates:
        token_str = " " + c.display.capitalize()
        ids = handle.tokenizer.encode(token_str, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"Empty tokenization for {token_str!r}")
        out[c.id] = ids[0]
    return out


def _find_unembed_path(model):
    """Return (final_norm_module, lm_head_module). Tries common Gemma 3
    paths for the final RMSNorm; lm_head via standard HF API."""
    norm = None
    for path in ("model.norm", "model.language_model.norm",
                 "language_model.model.norm", "language_model.norm"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            norm = obj
            break
    if norm is None:
        raise RuntimeError("Could not locate final norm module for logit lens")
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = model.lm_head
    return norm, lm_head


@torch.no_grad()
def _logit_lens_at_position(
    prefill_hidden_states: tuple[torch.Tensor, ...],
    position: int,
    norm,
    lm_head,
    class_token_ids: dict[str, int],
) -> tuple[torch.Tensor, list[str]]:
    """Apply final-norm + lm_head to the residual at `position` for every
    layer. Returns (n_layers, n_classes) tensor (CPU float32) of logits
    over the supplied class_token_ids, plus the ordered class id list.
    """
    n_layers = len(prefill_hidden_states)
    class_ids_list = list(class_token_ids.keys())
    device = prefill_hidden_states[0].device
    token_ids = torch.tensor(
        [class_token_ids[c] for c in class_ids_list], device=device
    )
    out = torch.zeros((n_layers, len(class_ids_list)), dtype=torch.float32)
    for L in range(n_layers):
        residual = prefill_hidden_states[L][0, position, :]  # (hidden,)
        normed = norm(residual.unsqueeze(0)).squeeze(0)  # (hidden,)
        logits_full = lm_head(normed)  # (vocab,)
        out[L] = logits_full[token_ids].detach().to("cpu", dtype=torch.float32)
    return out, class_ids_list


@torch.no_grad()
def _generate_reveal_greedy(
    handle: ModelHandle,
    model_inputs: dict[str, torch.Tensor],
    norm=None,
    lm_head=None,
    class_token_ids: dict[str, int] | None = None,
    max_new_tokens: int = 48,
) -> tuple[str, torch.Tensor | None, list[str] | None]:
    """Greedy reveal + optional logit-lens at pre_reveal_gen across all layers."""
    do_lens = norm is not None and lm_head is not None and class_token_ids
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": handle.tokenizer.eos_token_id,
        "do_sample": False,
    }
    if do_lens:
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs["output_hidden_states"] = True
    gen = handle.model.generate(**model_inputs, **gen_kwargs)
    if do_lens:
        seq_len = model_inputs["input_ids"].shape[1]
        new_tokens = gen.sequences[0, seq_len:]
        raw = handle.tokenizer.decode(new_tokens, skip_special_tokens=True)
        prefill_hs = gen.hidden_states[0]  # tuple per layer; each (1, seq_len, hidden)
        pre_reveal_pos = seq_len - 1
        lens_logits, class_ids_list = _logit_lens_at_position(
            prefill_hs, pre_reveal_pos, norm, lm_head, class_token_ids
        )
        return raw, lens_logits, class_ids_list
    new_tokens = gen[0, model_inputs["input_ids"].shape[1]:]
    raw = handle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw, None, None


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    bank = load_bank()
    turns_to_flip = [int(x) for x in args.turns_to_flip.split(",") if x.strip()]
    if not all(1 <= n <= 4 for n in turns_to_flip):
        print(f"--turns-to-flip must contain values in 1..4, got {turns_to_flip}", file=sys.stderr)
        return 2

    manifests = _load_kept_manifests(run_dir)
    print(f"Found {len(manifests)} kept runs in {run_dir}")

    # Subsample to first n_per_class per realized class
    by_class: dict[str, list[RunManifest]] = {}
    for m in manifests:
        by_class.setdefault(m.reveal_canonical_id, []).append(m)
    realized = sorted(by_class.keys())
    print(f"Realized classes: {realized}")
    selected: list[RunManifest] = []
    for c in realized:
        selected.extend(by_class[c][: args.n_per_class])
    print(f"Subsample (n_per_class={args.n_per_class}): {len(selected)} runs")
    print(f"Turns to flip: {turns_to_flip}")

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    handle = load_model(args.model, device=args.device, dtype=dtype)

    norm = lm_head = class_token_ids = None
    class_ids_list: list[str] = []
    if args.logit_lens:
        norm, lm_head = _find_unembed_path(handle.model)
        class_token_ids = _bank_first_token_ids(handle, bank)
        class_ids_list = list(class_token_ids.keys())
        print(f"Logit-lens enabled. Tracking {len(class_ids_list)} bank classes.")

    # ---- Per-run baseline reveal via fresh replay (no flip), for sanity ----
    baselines: list[dict] = []
    t0 = time.time()
    for m in selected:
        inputs = _build_full_reveal_inputs(handle, m, list(m.turns), bank, args.prompt_variant)
        raw, lens, _ = _generate_reveal_greedy(
            handle, inputs, norm=norm, lm_head=lm_head, class_token_ids=class_token_ids
        )
        canon = parse_reveal_to_canonical(raw, bank)
        rec = {
            "run_id": m.run_id,
            "class": m.reveal_canonical_id,
            "original_pattern": [t.answer_bool for t in m.turns],
            "baseline_replay_canonical": canon,
            "baseline_replay_raw": raw.strip(),
            "ondisk_canonical": m.reveal_canonical_id,
        }
        if lens is not None:
            rec["lens_logits"] = lens.tolist()  # (n_layers, n_classes)
        baselines.append(rec)
    print(f"Baselines: {len(baselines)} runs in {time.time()-t0:.1f}s")

    # ---- Flipped-turn trials ----
    trials: list[dict] = []
    t0 = time.time()
    total = len(selected) * len(turns_to_flip)
    i = 0
    for m in selected:
        for n in turns_to_flip:
            i += 1
            modified_turns, orig_ans, flipped_ans = _flip_turn_n_answer(m, n)
            if not flipped_ans:
                trials.append({
                    "run_id": m.run_id, "class": m.reveal_canonical_id,
                    "flipped_turn": n, "skipped": True,
                    "reason": f"unparseable original answer: {orig_ans!r}",
                })
                continue
            inputs = _build_full_reveal_inputs(handle, m, modified_turns, bank, args.prompt_variant)
            raw, lens, _ = _generate_reveal_greedy(
                handle, inputs, norm=norm, lm_head=lm_head, class_token_ids=class_token_ids
            )
            canon = parse_reveal_to_canonical(raw, bank)
            orig_pattern = [t.answer_bool for t in m.turns]
            flipped_pattern = list(orig_pattern)
            flipped_pattern[n - 1] = not flipped_pattern[n - 1] if flipped_pattern[n - 1] is not None else None
            rec = {
                "run_id": m.run_id,
                "class": m.reveal_canonical_id,
                "flipped_turn": n,
                "original_answer_text": orig_ans,
                "flipped_answer_text": flipped_ans,
                "original_pattern": orig_pattern,
                "flipped_pattern": flipped_pattern,
                "flipped_canonical": canon,
                "flipped_raw": raw.strip(),
            }
            if lens is not None:
                rec["lens_logits"] = lens.tolist()  # (n_layers, n_classes)
            trials.append(rec)
            if i % 40 == 0 or i == total:
                print(f"  [{i}/{total}] {m.reveal_canonical_id}/{m.run_id} "
                      f"flip_t{n}: {orig_ans}->{flipped_ans} -> {canon}", flush=True)
    print(f"Flipped trials: {len(trials)} in {time.time()-t0:.1f}s")

    # ---- Per-class × per-flipped-turn summary ----
    summary: dict[str, dict] = {}
    for cls in realized:
        for n in turns_to_flip:
            cell = [t for t in trials
                    if t.get("class") == cls
                    and t.get("flipped_turn") == n
                    and not t.get("skipped")]
            if not cell:
                continue
            kept_class = sum(1 for t in cell if t["flipped_canonical"] == cls) / len(cell)
            unparsed = sum(1 for t in cell if t["flipped_canonical"] is None)
            dist: dict[str, int] = {}
            for t in cell:
                k = t["flipped_canonical"] or "__unparsed__"
                dist[k] = dist.get(k, 0) + 1
            summary[f"{cls}/flip_t{n}"] = {
                "n": len(cell),
                "kept_class_rate": kept_class,
                "unparsed": unparsed,
                "distribution": dist,
            }

    out = {
        "run_dir": str(run_dir),
        "model": args.model,
        "torch_dtype": args.dtype,
        "prompt_variant": args.prompt_variant,
        "n_per_class": args.n_per_class,
        "turns_to_flip": turns_to_flip,
        "realized_classes": realized,
        "logit_lens_enabled": args.logit_lens,
        "logit_lens_class_order": class_ids_list if args.logit_lens else None,
        "logit_lens_class_token_ids": class_token_ids if args.logit_lens else None,
        "baselines": baselines,
        "trials": trials,
        "summary": summary,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json}")

    # Console summary table
    print()
    print(f"Kept-class rate matrix (rows=class, cols=turn flipped):")
    header = "  class    " + " | ".join(f"  T{n}  " for n in turns_to_flip)
    print(header)
    for cls in realized:
        row = [f"  {cls:8s}"]
        for n in turns_to_flip:
            s = summary.get(f"{cls}/flip_t{n}")
            if s is None:
                row.append("  --  ")
            else:
                row.append(f"{s['kept_class_rate']*100:5.1f}%")
        print(" | ".join(row))

    print()
    print("Output distributions per (class, flipped_turn):")
    for k in sorted(summary.keys()):
        s = summary[k]
        print(f"  {k:20s}  n={s['n']:2d} kept={s['kept_class_rate']*100:5.1f}%  dist={s['distribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
