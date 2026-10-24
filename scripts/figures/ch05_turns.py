"""A7 wrapper: regenerate the chapter 5-7 backfill plots (including ch05_turns.svg)."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
DATA_SOURCE = "见 scripts/figures/ch05_07_real_data.py（data/ch05_token_boundary.json、ch06_lora_scale.json、ch07_sft_metrics.json）"
ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "assets/images/plots")
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location("ch05_07_real_data", ROOT / "scripts/figures/ch05_07_real_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.render(args.out)


if __name__ == "__main__":
    main()
