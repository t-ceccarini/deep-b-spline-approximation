# -*- coding: utf-8 -*-
"""Inspect predicted parametrizations against classical baselines and true sampling t."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_b_spline_approximation.parametrization import (
    computeCentripetalParametrization,
    computeChordLengthParametrization,
    computeUniformParametrization,
)
from deep_b_spline_approximation.reproducibility import set_global_seed
from deep_b_spline_approximation.synthetic_dataset import generate_synthetic_dataset
from scripts.evaluate_ppn_mlp import build_ppn_inputs, build_ppn_model


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect parametrization quality for PPN and classical baselines.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-internal-knots", type=int, default=7)
    parser.add_argument("--non-uniform", action="store_true")
    parser.add_argument("--point-noise-std", type=float, default=0.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def summarize_params(name, predicted_t, true_t):
    delta = predicted_t[:, 1:] - predicted_t[:, :-1]
    min_delta = delta.min(dim=1).values
    max_delta = delta.max(dim=1).values

    true_centered = true_t - true_t.mean(dim=1, keepdim=True)
    pred_centered = predicted_t - predicted_t.mean(dim=1, keepdim=True)
    corr_num = torch.sum(true_centered * pred_centered, dim=1)
    corr_den = torch.sqrt(torch.sum(true_centered**2, dim=1) * torch.sum(pred_centered**2, dim=1)).clamp_min(1e-12)
    corr = corr_num / corr_den

    return {
        "name": name,
        "mae_to_true_t": torch.mean(torch.abs(predicted_t - true_t)).item(),
        "mean_min_delta": torch.mean(min_delta).item(),
        "mean_max_delta": torch.mean(max_delta).item(),
        "monotonicity_violation_rate": torch.mean((min_delta < -1e-10).to(torch.float64)).item(),
        "near_duplicate_step_rate": torch.mean((min_delta < 1e-8).to(torch.float64)).item(),
        "mean_corr_to_true_t": torch.mean(corr).item(),
    }


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesto ma non disponibile.")
    device = torch.device(args.device)

    data_internal_knots = None
    if args.data_internal_knots > 0:
        data_internal_knots = torch.linspace(0.0, 1.0, args.data_internal_knots + 2, dtype=torch.float64)[1:-1].tolist()

    _, curves, _, true_t = generate_synthetic_dataset(
        n_curves=args.n_eval,
        n_points=args.n_points,
        degree=3,
        internal_knots=data_internal_knots,
        non_uniform=args.non_uniform,
        point_noise_std=args.point_noise_std,
        seed=args.seed,
        return_params=True,
    )
    flat_curves = build_ppn_inputs(curves, args.checkpoint)

    ppn_model = build_ppn_model(
        checkpoint_path=args.checkpoint,
        n_points=args.n_points,
        hidden_size=args.hidden_size,
        device=device,
    )
    with torch.no_grad():
        _, ppn_t, _ = ppn_model(flat_curves.to(device))

    baseline_params = {
        "ppn_mlp": ppn_t.cpu(),
        "uniform": computeUniformParametrization(curves).cpu(),
        "chord_length": computeChordLengthParametrization(curves).cpu(),
        "centripetal": computeCentripetalParametrization(curves).cpu(),
    }

    results = {
        "dataset": {
            "n_eval": args.n_eval,
            "n_points": args.n_points,
            "data_internal_knots": args.data_internal_knots,
            "non_uniform": args.non_uniform,
            "point_noise_std": args.point_noise_std,
        },
        "parametrizations": {
            name: summarize_params(name, params, true_t)
            for name, params in baseline_params.items()
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"diagnostics_json={args.output_json}")


if __name__ == "__main__":
    main()
