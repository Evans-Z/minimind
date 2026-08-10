import argparse
import json
import os
import shutil
import subprocess


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "lm_harness_eval.json")
DEFAULT_TASKS = "ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa"
DEFAULT_MODELS = {
    "dense_64m_base": "hf_models/dense_64m",
    "dense_64m_mhc": "hf_models/mhc_64m_balm",
    "moe_198m_a64m_base": "hf_models/moe_198m_a64m",
    "moe_198m_a64m_mhc": "hf_models/mhc_moe_198m_a64m_balm",
}


def project_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_config(path):
    if not path:
        return {}
    path = project_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def value_from(args, config, key, default):
    value = getattr(args, key, None)
    if value is not None:
        return value
    return config.get(key, default)


def normalize_tasks(tasks):
    if isinstance(tasks, list):
        return ",".join(tasks)
    return str(tasks)


def parse_model_specs(specs, hf_root):
    if specs is None:
        return {
            name: os.path.join(PROJECT_ROOT, rel_path)
            for name, rel_path in DEFAULT_MODELS.items()
        }
    if isinstance(specs, dict):
        return {
            name: path if os.path.isabs(path) else os.path.join(hf_root, path)
            for name, path in specs.items()
        }

    models = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --models entry {spec!r}; expected name=path")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --models entry {spec!r}; expected name=path")
        if not os.path.isabs(path):
            path = os.path.join(hf_root, path)
        models[name] = path
    return models


def filter_selected_models(models, selected_models):
    if not selected_models:
        return models
    if isinstance(selected_models, str):
        selected_models = [selected_models]
    missing = [name for name in selected_models if name not in models]
    if missing:
        raise ValueError(
            f"selected_models contains unknown model(s): {', '.join(missing)}. "
            f"Available: {', '.join(models.keys())}"
        )
    return {name: models[name] for name in selected_models}


def parse_args():
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness on MiniMind HF exports.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="JSON config path. Use an empty string to disable.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model specs as name=path. Relative paths are resolved under --hf_root.",
    )
    parser.add_argument("--hf_root", default=None, help="Root for relative model paths")
    parser.add_argument("--tasks", default=None, help="Comma-separated lm-harness tasks")
    parser.add_argument("--batch_size", default=None, help="lm-harness batch size, e.g. 16 or auto")
    parser.add_argument("--device", default=None, help="Evaluation device, e.g. cuda:0, cpu, mps")
    parser.add_argument("--dtype", default=None, help="Model dtype passed to lm-harness")
    parser.add_argument("--output_dir", default=None, help="Where JSON results are saved")
    parser.add_argument("--selected_models", nargs="*", default=None, help="Run only these model names from the config")
    parser.add_argument(
        "--apply_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use only for chat/SFT models, not base models",
    )
    parser.add_argument("--limit", default=None, help="Optional lm-harness limit for smoke tests")
    parser.add_argument("--num_fewshot", default=None, help="Optional few-shot count")
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=None, help="Print commands without running them")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    hf_root = project_path(value_from(args, config, "hf_root", "hf_models"))
    tasks = normalize_tasks(value_from(args, config, "tasks", DEFAULT_TASKS))
    batch_size = str(value_from(args, config, "batch_size", "16"))
    device = str(value_from(args, config, "device", "cuda:0"))
    dtype = str(value_from(args, config, "dtype", "float16"))
    output_dir = project_path(value_from(args, config, "output_dir", "eval_results"))
    apply_chat_template = bool(value_from(args, config, "apply_chat_template", False))
    limit = str(value_from(args, config, "limit", "") or "")
    num_fewshot = str(value_from(args, config, "num_fewshot", "") or "")
    dry_run = bool(value_from(args, config, "dry_run", False))
    model_specs = args.models if args.models is not None else config.get("models")
    selected_models = value_from(args, config, "selected_models", [])

    lm_eval = shutil.which("lm_eval")
    if lm_eval is None and not dry_run:
        raise RuntimeError(
            "lm_eval not found. Install it first, for example: "
            "git clone https://github.com/EleutherAI/lm-evaluation-harness && "
            "cd lm-evaluation-harness && pip install -e ."
        )
    lm_eval = lm_eval or "lm_eval"

    models = filter_selected_models(parse_model_specs(model_specs, hf_root), selected_models)
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    for name, model_path in models.items():
        if not dry_run and not os.path.isdir(model_path):
            raise FileNotFoundError(f"HF model directory not found for {name}: {model_path}")

        output_path = os.path.join(output_dir, f"{name}.json")
        model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
        cmd = [
            lm_eval,
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            tasks,
            "--batch_size",
            batch_size,
            "--device",
            device,
            "--output_path",
            output_path,
        ]
        if apply_chat_template:
            cmd.append("--apply_chat_template")
        if limit:
            cmd.extend(["--limit", limit])
        if num_fewshot:
            cmd.extend(["--num_fewshot", num_fewshot])

        print("Running:", " ".join(cmd), flush=True)
        if dry_run:
            continue
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
