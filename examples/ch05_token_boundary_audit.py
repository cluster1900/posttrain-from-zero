"""Audit piecewise vs whole-text tokenisation on the real SFT trajectory set.

Reader experiment that produced ``data/ch05_token_boundary.json``.  The pure
helpers (``piecewise_ids``, ``crossed_boundaries``, ``summarize``) run offline
and are unit-tested with a toy tokenizer.  ``main`` is OPTIONAL and needs:

* a checkout of the source project at commit ca57456 (``--project``), whose
  ``train/dataset.py`` builds the real SFT examples;
* the public SFT parquet at revision 9d6ab74 (``--parquet``; SHA-256 must be
  7507b499…e38a5, the value recorded in the round1 data info file);
* network access for the Qwen3.5-2B-Base tokenizer at revision b1485b2.

    python examples/ch05_token_boundary_audit.py --project ../src --parquet train.parquet
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARQUET_SHA256 = "7507b499a0e9289086bb4befa4344f81c5d2ca91698e2ea22b81cc60834e38a5"
TOKENIZER = ("Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
LENGTH_LIMITS = (16384, 24576, 32768)


def piecewise_ids(encode, text: str, cuts: list[int]) -> list[int]:
    """Encode text[cuts[i]:cuts[i+1]] separately and concatenate (the project SFT rule)."""
    edges = [0] + sorted(c for c in set(cuts) if 0 < c < len(text)) + [len(text)]
    ids: list[int] = []
    for a, b in zip(edges, edges[1:]):
        ids.extend(encode(text[a:b]))
    return ids


def crossed_boundaries(offsets: list[tuple[int, int]], cuts: list[int]) -> list[int]:
    """Cuts that fall strictly inside one whole-text token (that token straddles the cut)."""
    return [c for c in sorted(set(cuts)) if any(a < c < e for a, e in offsets)]


def summarize(records: list[dict]) -> dict:
    """Aggregate per-trajectory audit records into the committed JSON summary."""
    n = len(records)
    same = sum(r["piece_ids_equal_whole"] for r in records)
    turns = collections.Counter(r["assistant_turns"] for r in records)
    delta = collections.Counter(r["len_piece"] - r["len_whole"] for r in records)
    stats = collections.Counter()
    for r in records:
        stats.update(r["stats"])
    return {
        "trajectories": n,
        "reasons": dict(collections.Counter(r["reason"] for r in records)),
        "piece_equal_whole": same,
        "piece_differs_from_whole": n - same,
        "span_boundaries": sum(r["boundaries"] for r in records),
        "crossed_boundaries": sum(r["crossed"] for r in records),
        "crossed_at_span_start": sum(r["crossed_start"] for r in records),
        "delta_equals_crossed_for_all": all(r["len_piece"] - r["len_whole"] == r["crossed"] for r in records),
        "total_tokens_piecewise": sum(r["len_piece"] for r in records),
        "total_tokens_whole": sum(r["len_whole"] for r in records),
        "total_label_tokens": sum(r["label_tokens"] for r in records),
        "retained_at": {str(L): sum(r["len_piece"] <= L for r in records) for L in LENGTH_LIMITS},
        "assistant_turns_hist": {str(k): turns[k] for k in sorted(turns)},
        "extra_tokens_hist": {str(k): delta[k] for k in sorted(delta)},
        "turn_stats": dict(sorted(stats.items())),
    }


def main(argv=None):  # pragma: no cover - needs network, GPU-free but large
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data/ch05_token_boundary.json")
    args = ap.parse_args(argv)
    digest = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    if digest != PARQUET_SHA256:
        raise SystemExit(f"parquet sha256 mismatch: {digest}")
    sys.path.insert(0, str(args.project))
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from train import dataset as D
    from train.chat_format import apply_agent_chat_template

    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    apply_agent_chat_template(tok)
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    records = []
    for i, row in enumerate(pq.read_table(args.parquet).to_pylist()):
        st = collections.Counter()
        ex, reason = D.build_sft_example(tok, row["messages"], tools=row["tools"], max_seq_len=10**9, stats=st)
        msgs = D.normalize_messages(row["messages"])
        last = max(k for k, m in enumerate(msgs) if m["role"] == "assistant")
        text = tok.apply_chat_template(msgs[: last + 1], tools=D.parse_tools(row["tools"]), tokenize=False)
        spans = D.assistant_spans(text)
        cuts = [x for s in spans for x in (s.start, s.end)]
        starts = {s.start for s in spans}
        whole = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        crossed = crossed_boundaries(whole["offset_mapping"], cuts)
        assert ex is not None and piecewise_ids(encode, text, cuts) == ex.input_ids
        records.append({
            "reason": reason,
            "assistant_turns": sum(m["role"] == "assistant" for m in row["messages"]),
            "stats": dict(st),
            "len_piece": len(ex.input_ids),
            "len_whole": len(whole["input_ids"]),
            "piece_ids_equal_whole": whole["input_ids"] == ex.input_ids,
            "label_tokens": sum(l != -100 for l in ex.labels),
            "boundaries": len({c for c in cuts if 0 < c < len(text)}),
            "crossed": len(crossed),
            "crossed_start": sum(c in starts for c in crossed),
        })
    pre = "<|im_start|>assistant\n<think>\n"
    gen = "\n</think>\n\n<tool_call>"
    summary = {
        "source": {
            "project_commit": "ca57456",
            "project_files": ["train/dataset.py", "train/chat_format.py", "configs/chat_templates/qwen35_agent.jinja"],
            "dataset_revision": "9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4",
            "parquet_sha256": PARQUET_SHA256,
            "tokenizer_revision": TOKENIZER[1],
            "kind": "reader_experiment_on_real_project_data",
        },
        **summarize(records),
        "minimal_example": {
            "prefix_ids": encode(pre),
            "generated_ids": encode(gen),
            "whole_ids": encode(pre + gen),
        },
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
