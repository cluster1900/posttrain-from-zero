from pathlib import Path
import runpy

DATA_SOURCE = "fixed experiment commit ca57456: result/round2/results/rl_2b_24k_metrics.jsonl"

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("ch09_rl_2b_24k.py")), run_name="__main__")
