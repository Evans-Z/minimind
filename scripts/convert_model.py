import argparse
import json
import os
import sys
import warnings

__package__ = "scripts"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_minimind_mhc import MiniMindMHCConfig, MiniMindMHCForCausalLM
from model.model_lora import apply_lora, merge_lora

warnings.filterwarnings('ignore', category=UserWarning)


def _dtype_from_name(name):
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_yaml(path):
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("PyYAML is required for preset-based conversion. Please install: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_preset(config_yaml, size_preset="", context_preset=""):
    resolved = {}
    if not config_yaml:
        return resolved
    data = _load_yaml(config_yaml)
    if size_preset:
        size_presets = data.get("size_presets") or {}
        if size_preset not in size_presets:
            raise ValueError(f"size preset not found: {size_preset}. Available: {', '.join(sorted(size_presets))}")
        resolved.update(size_presets[size_preset])
    if context_preset:
        context_presets = data.get("context_presets") or {}
        if context_preset not in context_presets:
            raise ValueError(f"context preset not found: {context_preset}. Available: {', '.join(sorted(context_presets))}")
        resolved.update(context_presets[context_preset])
    return resolved


def _coerce_bool_int(config, key):
    if key in config:
        config[key] = bool(config[key])


def build_minimind_config(config_kwargs):
    config_kwargs = dict(config_kwargs)
    model_variant = config_kwargs.pop("model_variant", "minimind")
    _coerce_bool_int(config_kwargs, "use_moe")
    _coerce_bool_int(config_kwargs, "hc_overlap")
    if model_variant == "mhc":
        return MiniMindMHCConfig(**config_kwargs), model_variant
    if model_variant == "minimind":
        return MiniMindConfig(**config_kwargs), model_variant
    raise ValueError(f"Unsupported model_variant={model_variant!r}")


def infer_torch_path(save_dir, weight, lm_config, model_variant):
    variant_suffix = f"_{model_variant}" if model_variant else ""
    moe_suffix = "_moe" if lm_config.use_moe else ""
    primary = os.path.join(save_dir, f"{weight}_{lm_config.hidden_size}{variant_suffix}{moe_suffix}.pth")
    legacy = os.path.join(save_dir, f"{weight}_{lm_config.hidden_size}{moe_suffix}.pth")
    if os.path.exists(primary):
        return primary
    if model_variant == "minimind" and os.path.exists(legacy):
        return legacy
    return primary


def _patch_tokenizer_config(transformers_path):
    tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        tokenizer_config.update({"tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}})
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)


def convert_torch2hf_custom(torch_path, transformers_path, lm_config, model_variant="minimind", tokenizer_path=None, dtype=torch.float16):
    if model_variant == "mhc":
        MiniMindMHCConfig.register_for_auto_class()
        MiniMindMHCForCausalLM.register_for_auto_class("AutoModelForCausalLM")
        lm_model = MiniMindMHCForCausalLM(lm_config)
    else:
        MiniMindConfig.register_for_auto_class()
        MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
        lm_model = MiniMindForCausalLM(lm_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(torch_path, map_location=device)
    missing, unexpected = lm_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys while loading: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"Warning: unexpected keys while loading: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    lm_model = lm_model.to(dtype)
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f"模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)")
    lm_model.save_pretrained(transformers_path, safe_serialization=False)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or os.path.join(PROJECT_ROOT, "model"))
    tokenizer.save_pretrained(transformers_path)
    _patch_tokenizer_config(transformers_path)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")

def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# QwenForCausalLM/LlamaForCausalLM结构兼容生态
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": lm_config.tie_word_embeddings
    }
    if not lm_config.use_moe:
        qwen_config = Qwen3Config(
            **common_config, 
            use_sliding_window=False, 
            sliding_window=None
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)
        # ======= transformers-5.0的兼容低版本写法 =======
        if int(transformers.__version__.split('.')[0]) >= 5:
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                new_sd[f'{p}.gate_up_proj'] = torch.cat([torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]), torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])], dim=1)
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype)  # 转换模型权重精度
    qwen_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    apply_lora(lm_model)
    merge_lora(lm_model, lora_path, merged_torch_path)
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")


def convert_jinja_to_json(jinja_path):
    with open(jinja_path, 'r') as f: template = f.read()
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')


def convert_json_to_jinja(json_file_path, output_path):
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind model conversion utilities")
    subparsers = parser.add_subparsers(dest="command")

    torch2hf = subparsers.add_parser(
        "torch2hf_custom",
        help="Convert native MiniMind/MHC .pth weights to custom Transformers format for lm-harness.",
    )
    torch2hf.add_argument("--model_config_yaml", default="", type=str, help="YAML file containing size/context presets")
    torch2hf.add_argument("--size_preset", default="", type=str, help="Preset name under size_presets")
    torch2hf.add_argument("--context_preset", default="", type=str, help="Preset name under context_presets")
    torch2hf.add_argument("--model_variant", default="", choices=["", "minimind", "mhc"], help="Override model variant")
    torch2hf.add_argument("--hidden_size", default=0, type=int, help="Override hidden size")
    torch2hf.add_argument("--num_hidden_layers", default=0, type=int, help="Override hidden layer count")
    torch2hf.add_argument("--use_moe", default=-1, type=int, choices=[-1, 0, 1], help="Override MoE flag")
    torch2hf.add_argument("--save_dir", default=os.path.join(PROJECT_ROOT, "out"), type=str, help="Directory containing .pth weights")
    torch2hf.add_argument("--weight", default="pretrain_scale", type=str, help="Weight prefix, e.g. pretrain_scale")
    torch2hf.add_argument("--torch_path", default="", type=str, help="Explicit .pth path; overrides inferred path")
    torch2hf.add_argument("--output_dir", default="", type=str, help="Output Transformers model directory")
    torch2hf.add_argument("--tokenizer_path", default=os.path.join(PROJECT_ROOT, "model"), type=str, help="Tokenizer directory")
    torch2hf.add_argument("--dtype", default="float16", type=str, help="float16, bfloat16, or float32")

    return parser.parse_args()


def run_torch2hf_custom(args):
    config_kwargs = _apply_preset(args.model_config_yaml, args.size_preset, args.context_preset)
    if args.model_variant:
        config_kwargs["model_variant"] = args.model_variant
    if args.hidden_size:
        config_kwargs["hidden_size"] = args.hidden_size
    if args.num_hidden_layers:
        config_kwargs["num_hidden_layers"] = args.num_hidden_layers
    if args.use_moe != -1:
        config_kwargs["use_moe"] = args.use_moe
    if not config_kwargs:
        config_kwargs = {"model_variant": "minimind", "hidden_size": 768, "num_hidden_layers": 8, "use_moe": False}

    lm_config, model_variant = build_minimind_config(config_kwargs)
    torch_path = args.torch_path or infer_torch_path(args.save_dir, args.weight, lm_config, model_variant)
    if not os.path.exists(torch_path):
        raise FileNotFoundError(f"Checkpoint not found: {torch_path}")

    output_dir = args.output_dir
    if not output_dir:
        preset_name = args.size_preset or f"{model_variant}_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}"
        output_dir = os.path.join(PROJECT_ROOT, "hf_models", preset_name)

    convert_torch2hf_custom(
        torch_path=torch_path,
        transformers_path=output_dir,
        lm_config=lm_config,
        model_variant=model_variant,
        tokenizer_path=args.tokenizer_path,
        dtype=_dtype_from_name(args.dtype),
    )


if __name__ == '__main__':
    cli_args = parse_args()
    if cli_args.command == "torch2hf_custom":
        run_torch2hf_custom(cli_args)
        raise SystemExit(0)

    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # convert torch to transformers
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3'
    convert_torch2transformers(torch_path, transformers_path)

    # # merge lora
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
