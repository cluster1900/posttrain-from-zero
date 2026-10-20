"""OPTIONAL online config-only inspection, outside CI. Never downloads weights.

Run explicitly from repository root: python -m examples.ch06_official_config_optional
This is a reader experiment, not a claim about the original project's measurements.
"""
import json
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5Config
from examples.ch06_model_structure import inspect_model

MODEL = "Qwen/Qwen3.5-2B-Base"
REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def main():
    path = hf_hub_download(MODEL, "config.json", revision=REVISION)
    config = Qwen3_5Config.from_dict(json.loads(open(path, encoding="utf-8").read()))
    _, report = inspect_model(config.text_config, meta=True)
    report["scope"] = "text backbone + tied output head; vision tower excluded; no weights loaded"
    report["config_revision"] = REVISION
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
