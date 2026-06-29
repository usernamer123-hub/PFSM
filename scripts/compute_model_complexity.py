import argparse
import importlib
import inspect
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================
# Direct-run configuration
# =========================
# Set RUN_FROM_CONFIG = True and run:
#   python E:\桌面\a2\scripts\compute_model_complexity.py
RUN_FROM_CONFIG = False

CONFIG_MODELS = [
    # Built-in project model.
    {
        "name": "PFSM",
        "target": "nets.pfsm:PFSMNet",
        "kwargs": {"in_channels": 12, "out_channels": 12, "img_size": 512},
        "input_shape": (1, 12, 512, 512),
    },
    # Add comparison methods here if their code is available.
    # Example:
    # {
    #     "name": "YourMethod",
    #     "target": "some_package.some_file:SomeModel",
    #     "kwargs": {"in_channels": 3, "out_channels": 3},
    #     "input_shape": (1, 3, 512, 512),
    # },
]

CONFIG_OUTPUT_CSV = str(ROOT / "complexity_results.csv")
CONFIG_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def import_target(target):
    if ":" not in target:
        raise ValueError("target must use the format module.path:ClassName")
    module_name, attr_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def instantiate_model(spec):
    cls_or_fn = import_target(spec["target"])
    kwargs = dict(spec.get("kwargs", {}))
    model = cls_or_fn(**kwargs)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_profile_thop(model, input_shape, device):
    try:
        from thop import profile
    except Exception as exc:
        return None, f"thop is not available: {exc}"

    dummy = torch.randn(*input_shape, device=device)
    try:
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops, None
    except Exception as exc:
        return None, str(exc)


def format_million(value):
    return value / 1e6


def format_billion(value):
    return value / 1e9


def run_specs(specs, output_csv=None, device="cpu"):
    rows = []
    for spec in specs:
        name = spec["name"]
        input_shape = tuple(spec.get("input_shape", (1, 3, 512, 512)))
        print(f"\n==> Profiling {name}")
        print(f"    target: {spec['target']}")
        print(f"    input : {input_shape}")

        try:
            model = instantiate_model(spec).to(device)
            model.eval()
        except Exception as exc:
            print(f"    ERROR: failed to instantiate model: {exc}")
            rows.append(
                {
                    "Method": name,
                    "Input": str(input_shape),
                    "Params(M)": "",
                    "Trainable(M)": "",
                    "FLOPs(G)": "",
                    "Status": f"instantiate failed: {exc}",
                }
            )
            continue

        total, trainable = count_params(model)
        flops, flops_err = try_profile_thop(model, input_shape, device)

        row = {
            "Method": name,
            "Input": "x".join(map(str, input_shape)),
            "Params(M)": f"{format_million(total):.3f}",
            "Trainable(M)": f"{format_million(trainable):.3f}",
            "FLOPs(G)": f"{format_billion(flops):.3f}" if flops is not None else "",
            "Status": "OK" if flops_err is None else f"FLOPs failed: {flops_err}",
        }
        rows.append(row)

        print(f"    Params   : {row['Params(M)']} M")
        print(f"    Trainable: {row['Trainable(M)']} M")
        if flops is not None:
            print(f"    FLOPs    : {row['FLOPs(G)']} G")
        else:
            print(f"    FLOPs    : failed ({flops_err})")

    if output_csv:
        import csv

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Method", "Input", "Params(M)", "Trainable(M)", "FLOPs(G)", "Status"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {output_csv}")
    return rows


def parse_model_spec(value):
    # Format:
    # Name=module.path:ClassName
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use Name=module.path:ClassName")
    name, target = value.split("=", 1)
    name = name.strip()
    target = target.strip()
    if not name or not target:
        raise argparse.ArgumentTypeError("Name and target cannot be empty")
    return {"name": name, "target": target, "kwargs": {}}


def main():
    if RUN_FROM_CONFIG:
        run_specs(CONFIG_MODELS, output_csv=CONFIG_OUTPUT_CSV, device=CONFIG_DEVICE)
        return

    parser = argparse.ArgumentParser(description="Compute Params and FLOPs for comparison models.")
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        help="Model spec, e.g. PFSM=nets.pfsm:PFSMNet. Repeat for multiple models.",
    )
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default=CONFIG_DEVICE)
    parser.add_argument("--csv", default=str(ROOT / "complexity_results.csv"))
    args = parser.parse_args()

    specs = args.model or [
        {
            "name": "PFSM",
            "target": "nets.pfsm:PFSMNet",
            "kwargs": {
                "in_channels": args.input_channels,
                "out_channels": args.input_channels,
                "img_size": args.width,
            },
            "input_shape": (1, args.input_channels, args.height, args.width),
        }
    ]

    for spec in specs:
        spec.setdefault("kwargs", {})
        spec.setdefault("input_shape", (1, args.input_channels, args.height, args.width))

    run_specs(specs, output_csv=args.csv, device=args.device)


if __name__ == "__main__":
    main()

