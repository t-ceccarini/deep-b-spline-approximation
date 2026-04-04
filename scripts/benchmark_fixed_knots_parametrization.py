# -*- coding: utf-8 -*-
"""Benchmark parametrizations with a shared fixed B-spline knot vector."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_b_spline_approximation.BSpline import computeBSplineApproximation
from deep_b_spline_approximation.parametrization import (
    computeCentripetalParametrization,
    computeChordLengthParametrization,
    computeUniformParametrization,
)
from deep_b_spline_approximation.reproducibility import set_global_seed
from deep_b_spline_approximation.synthetic_dataset import generate_synthetic_dataset
from scripts.evaluate_ppn_mlp import build_ppn_inputs, build_ppn_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PPN/baseline parametrizations under a shared fixed knot vector."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-internal-knots", type=int, default=7)
    parser.add_argument("--data-internal-knots", type=int, default=7)
    parser.add_argument("--non-uniform", action="store_true")
    parser.add_argument("--point-noise-std", type=float, default=0.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def build_uniform_internal_knots(n_internal_knots, degree=3):
    if n_internal_knots <= 0:
        return torch.tensor([0.0] * (degree + 1) + [1.0] * (degree + 1), dtype=torch.float64)
    internal = np.linspace(0.0, 1.0, n_internal_knots + 2, dtype=np.float64)[1:-1]
    knots = np.concatenate(([0.0] * (degree + 1), internal, [1.0] * (degree + 1)))
    return torch.tensor(knots, dtype=torch.float64)


def compute_metrics(curves, splines):
    mse = torch.mean((curves - splines) ** 2).item()
    curves_np = curves.detach().cpu().numpy()
    splines_np = splines.detach().cpu().numpy()
    dhd = float(
        np.mean(
            [
                directed_hausdorff(curve, spline)[0]
                for curve, spline in zip(curves_np, splines_np)
            ]
        )
    )
    return {"mse": mse, "directed_hausdorff": dhd}


def evaluate_ppn_with_fixed_knots(model, flat_curves, curves, knots, device):
    model.eval()
    with torch.no_grad():
        _, params, _ = model(flat_curves.to(device))
        splines = computeBSplineApproximation(curves.to(device), params.to(device), device=device, u=knots.to(device))
    return compute_metrics(curves.to(device), splines)


def evaluate_baseline_with_fixed_knots(parametrization_fn, curves, knots, device):
    with torch.no_grad():
        params = parametrization_fn(curves.to(device))
        splines = computeBSplineApproximation(curves.to(device), params.to(device), device=device, u=knots.to(device))
    return compute_metrics(curves.to(device), splines)


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesto ma non disponibile.")
    device = torch.device(args.device)

    data_internal_knots = None
    if args.data_internal_knots > 0:
        data_internal_knots = np.linspace(0.0, 1.0, args.data_internal_knots + 2, dtype=np.float64)[1:-1].tolist()

    _, curves, _ = generate_synthetic_dataset(
        n_curves=args.n_eval,
        n_points=args.n_points,
        degree=3,
        internal_knots=data_internal_knots,
        non_uniform=args.non_uniform,
        point_noise_std=args.point_noise_std,
        seed=args.seed,
    )
    flat_curves = build_ppn_inputs(curves, args.checkpoint)
    knots = build_uniform_internal_knots(args.n_internal_knots, degree=3)

    ppn_model = build_ppn_model(
        checkpoint_path=args.checkpoint,
        n_points=args.n_points,
        hidden_size=args.hidden_size,
        device=device,
    )

    results = {
        "dataset": {
            "n_eval": args.n_eval,
            "n_points": args.n_points,
            "data_internal_knots": args.data_internal_knots,
            "fit_internal_knots": args.n_internal_knots,
            "non_uniform": args.non_uniform,
            "point_noise_std": args.point_noise_std,
        },
        "ppn_mlp": evaluate_ppn_with_fixed_knots(ppn_model, flat_curves, curves, knots, device),
        "uniform": evaluate_baseline_with_fixed_knots(computeUniformParametrization, curves, knots, device),
        "chord_length": evaluate_baseline_with_fixed_knots(computeChordLengthParametrization, curves, knots, device),
        "centripetal": evaluate_baseline_with_fixed_knots(computeCentripetalParametrization, curves, knots, device),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"benchmark_json={args.output_json}")


if __name__ == "__main__":
    main()
