# -*- coding: utf-8 -*-
"""P1-light benchmark for ID/OOD checks against classical parametrizations."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_b_spline_approximation.synthetic_dataset import generate_synthetic_dataset
from scripts.evaluate_ppn_mlp import build_ppn_inputs, build_ppn_model, FixedParametrizationModel
from deep_b_spline_approximation.parametrization import (
    computeCentripetalParametrization,
    computeChordLengthParametrization,
    computeUniformParametrization,
)
from deep_b_spline_approximation.directedHausdorff import computeMeanHausdorffDistance3
from deep_b_spline_approximation.reproducibility import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run a lightweight ID/OOD benchmark for PPN-MLP.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/ppn_mlp/benchmark_metrics.json"))
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def _evaluate_ppn(model, flat_curves, criterion):
    model.eval()
    with torch.no_grad():
        splines, _, curves = model(flat_curves.to(next(model.parameters()).device))
        mse = criterion(curves, splines).item()
        dhd = computeMeanHausdorffDistance3(flat_curves, model, k=3, p=2)
    return {"mse": mse, "directed_hausdorff": dhd}


def _evaluate_baseline(model, curves, criterion):
    model.eval()
    with torch.no_grad():
        splines, _, target = model(curves.to(model.device))
        mse = criterion(target, splines).item()
        dhd = computeMeanHausdorffDistance3(curves, model, k=3, p=2)
    return {"mse": mse, "directed_hausdorff": dhd}


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesto ma non disponibile.")
    device = torch.device(args.device)

    ppn_model = build_ppn_model(
        checkpoint_path=args.checkpoint,
        n_points=args.n_points,
        hidden_size=args.hidden_size,
        device=device,
    )
    baselines = {
        "uniform": FixedParametrizationModel(computeUniformParametrization, device=device),
        "chord_length": FixedParametrizationModel(computeChordLengthParametrization, device=device),
        "centripetal": FixedParametrizationModel(computeCentripetalParametrization, device=device),
    }
    scenarios = {
        "id_bezier_uniform": {
            "internal_knots": None,
            "non_uniform": False,
            "point_noise_std": 0.0,
            "sigma": (1.0, 2.0),
        },
        "ood_bezier_non_uniform": {
            "internal_knots": None,
            "non_uniform": True,
            "point_noise_std": 0.0,
            "sigma": (1.0, 2.0),
        },
        "ood_bezier_noise": {
            "internal_knots": None,
            "non_uniform": False,
            "point_noise_std": 0.05,
            "sigma": (1.0, 2.0),
        },
        "ood_bspline_internal_knots": {
            "internal_knots": [0.25, 0.5, 0.75],
            "non_uniform": False,
            "point_noise_std": 0.0,
            "sigma": (1.0, 2.0),
        },
    }

    criterion = nn.MSELoss()
    results = {}
    for scenario_name, dataset_kwargs in scenarios.items():
        flat_curves, curves, _ = generate_synthetic_dataset(
            n_curves=args.n_eval,
            n_points=args.n_points,
            degree=3,
            seed=args.seed,
            **dataset_kwargs,
        )
        model_inputs = build_ppn_inputs(curves, args.checkpoint)
        scenario_result = {"ppn_mlp": _evaluate_ppn(ppn_model, model_inputs, criterion)}
        for name, baseline_model in baselines.items():
            scenario_result[name] = _evaluate_baseline(baseline_model, curves, criterion)
        results[scenario_name] = scenario_result
        print(f"{scenario_name}: {json.dumps(scenario_result)}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"benchmark_json={args.output_json}")


if __name__ == "__main__":
    main()
