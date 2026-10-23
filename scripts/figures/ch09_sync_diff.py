"""A7 wrapper: regenerate all ch09 RL plots (including this stem)."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
DATA_SOURCE = "固定实验提交 ca57456：见 scripts/figures/ch09_rl_metrics.py"
ROOT = Path(__file__).resolve().parents[2]

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "assets/images/plots")
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location(
        "ch09_rl_metrics", ROOT / "scripts/figures/ch09_rl_metrics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.render(args.out)

if __name__ == "__main__":
    main()
