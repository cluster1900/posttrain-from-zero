"""Inspect a random, two-layer Qwen3.5 text model; never download weights.

Fixture dimensions follow tests/conftest.py at ca57456. Counts here are reader
experiments, not historical project measurements or the full 2B model size.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
    "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj",
)


def tiny_config():
    return Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        full_attention_interval=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_conv_kernel_dim=4, linear_key_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_value_head_dim=16,
        tie_word_embeddings=True, attn_output_gate=True, attention_bias=False,
        max_position_embeddings=32768, rms_norm_eps=1e-6,
        rope_parameters={"mrope_interleaved": True, "mrope_section": [1, 1, 0],
                         "rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": 0.25},
    )


def inspect_model(config=None, *, meta=False):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = config or tiny_config()
    with torch.device("meta" if meta else "cpu"):
        base = AutoModelForCausalLM.from_config(config)
    base_total = sum(p.numel() for p in base.parameters())
    embedding = base.get_input_embeddings().weight.numel()
    targets = []
    tree = []
    types = Counter()
    for name, module in base.named_modules():
        if name:
            tree.append({"name": name, "type": type(module).__name__})
            types[type(module).__name__] += 1
        if name.rsplit(".", 1)[-1] in TARGETS:
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"unexpected target type: {name}")
            targets.append({"name": name, "in": module.in_features,
                            "out": module.out_features,
                            "lora_parameters": 64 * (module.in_features + module.out_features)})
    expected = sum(row["lora_parameters"] for row in targets)
    model = get_peft_model(base, LoraConfig(
        r=64, lora_alpha=64, lora_dropout=0.0, target_modules=list(TARGETS),
        bias="none", task_type="CAUSAL_LM",
    ))
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trained == expected
    assert not model.get_input_embeddings().weight.requires_grad
    assert not model.get_output_embeddings().weight.requires_grad
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    leaves = {row["name"].rsplit(".", 1)[-1] for row in targets}
    assert leaves == set(TARGETS)
    report = {
        "kind": "reader_experiment_not_project_metrics",
        "model_class": type(base).__name__, "meta_only": meta,
        "base_unique_parameters": base_total, "embedding_unique_parameters": embedding,
        "lora_trainable_parameters": trained, "wrapped_unique_parameters": total,
        "trainable_fraction": trained / total, "target_instances": targets,
        "module_types": dict(sorted(types.items())), "module_tree": tree,
        "rank": 64, "alpha": 64, "dropout": 0.0,
        "layer_types": list(config.layer_types), "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
    }
    return model, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    _, report = inspect_model()
    for row in report["module_tree"]:
        print(f"{row['name']}: {row['type']}")
    print(json.dumps({k: v for k, v in report.items() if k != "module_tree"}, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
