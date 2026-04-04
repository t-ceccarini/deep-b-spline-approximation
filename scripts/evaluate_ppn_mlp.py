# -*- coding: utf-8 -*-
"""Minimal PPN-MLP evaluation with classical parametrization baselines."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_b_spline_approximation.BSpline import computeBSplineApproximation
from deep_b_spline_approximation.directedHausdorff import computeMeanHausdorffDistance3
from deep_b_spline_approximation.parametrization import (
    computeCentripetalParametrization,
    computeChordLengthParametrization,
    computeUniformParametrization,
)
from deep_b_spline_approximation.parametrization_features import (
    build_enriched_curve_inputs,
    parse_parametrization_feature_names,
)
from deep_b_spline_approximation.ppn import (
    CentripetalResidualParametrizationNetwork,
    ConvexMixtureParametrizationNetwork,
    PointParametrizationNetwork,
)
from deep_b_spline_approximation.reproducibility import set_global_seed
from deep_b_spline_approximation.synthetic_dataset import generate_synthetic_dataset, load_parnet_txt_dataset


class FixedParametrizationModel(nn.Module):
    def __init__(self, parametrization_fn, device="cpu"):
        super().__init__()
        self.parametrization_fn = parametrization_fn
        self.device = device

    def forward(self, curves):
        curves = curves.to(self.device)
        params = self.parametrization_fn(curves)
        splines = computeBSplineApproximation(curves, params, self.device)
        return splines, params, curves


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PPN-MLP checkpoint and classical baselines.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/ppn_mlp/eval_metrics.json"))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--non-uniform", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def build_ppn_model(checkpoint_path, n_points, hidden_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    input_size = int(config.get("n_points", n_points)) * 2
    model_hidden = int(config.get("hidden_size", hidden_size))
    n_fit_internal_knots = int(config.get("fit_internal_knots", 0) or 0)
    fit_internal_knots = None
    if n_fit_internal_knots > 0:
        fit_internal_knots = np.linspace(0.0, 1.0, n_fit_internal_knots + 2, dtype=np.float64)[1:-1].tolist()
    normalization = config.get("normalization", "batchnorm")
    param_input_names = parse_parametrization_feature_names(config.get("param_inputs", ""))
    model_kind = config.get("model_kind", "ppn_mlp")
    input_channels = 2 + len(param_input_names)

    if model_kind == "ppn_mlp":
        model = PointParametrizationNetwork(
            inputSize=int(config.get("n_points", n_points)) * input_channels,
            hiddenSize=model_hidden,
            p=2,
            k=3,
            device=device,
            internal_knots=fit_internal_knots,
            normalization=normalization,
            input_channels=input_channels,
        )
    elif model_kind == "baseline_mixture":
        model = ConvexMixtureParametrizationNetwork(
            inputSize=int(config.get("n_points", n_points)) * 2,
            hiddenSize=model_hidden,
            p=2,
            k=3,
            device=device,
            internal_knots=fit_internal_knots,
            normalization=normalization,
        )
    else:
        model = CentripetalResidualParametrizationNetwork(
            inputSize=int(config.get("n_points", n_points)) * 2,
            hiddenSize=model_hidden,
            p=2,
            k=3,
            device=device,
            internal_knots=fit_internal_knots,
            normalization=normalization,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_ppn_inputs(curves, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if config.get("model_kind", "ppn_mlp") != "ppn_mlp":
        return build_enriched_curve_inputs(curves, [])
    param_input_names = parse_parametrization_feature_names(config.get("param_inputs", ""))
    return build_enriched_curve_inputs(curves, param_input_names)


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesto ma non disponibile.")
    device = torch.device(args.device)

    if args.dataset_path is None:
        _, curves_eval, _ = generate_synthetic_dataset(
            n_curves=args.n_eval,
            n_points=args.n_points,
            degree=3,
            internal_knots=None,
            non_uniform=args.non_uniform,
            seed=args.seed,
        )
    else:
        _, curves_eval = load_parnet_txt_dataset(args.dataset_path, p=2)
    flat_eval = build_ppn_inputs(curves_eval, args.checkpoint)

    ppn_model = build_ppn_model(
        checkpoint_path=args.checkpoint,
        n_points=args.n_points,
        hidden_size=args.hidden_size,
        device=device,
    )

    baseline_models = {
        "uniform": FixedParametrizationModel(computeUniformParametrization, device=device),
        "chord_length": FixedParametrizationModel(computeChordLengthParametrization, device=device),
        "centripetal": FixedParametrizationModel(computeCentripetalParametrization, device=device),
    }

    criterion = nn.MSELoss()
    with torch.no_grad():
        pred_splines, _, pred_curves = ppn_model(flat_eval.to(device))
        ppn_mse = criterion(pred_curves, pred_splines).item()
        ppn_dhd = computeMeanHausdorffDistance3(flat_eval, ppn_model, k=3, p=2)

    results = {
        "ppn_mlp": {
            "mse": ppn_mse,
            "directed_hausdorff": ppn_dhd,
        }
    }

    for name, model in baseline_models.items():
        with torch.no_grad():
            splines, _, curves = model(curves_eval.to(device))
            mse = criterion(curves, splines).item()
            dhd = computeMeanHausdorffDistance3(curves_eval, model, k=3, p=2)
        results[name] = {
            "mse": mse,
            "directed_hausdorff": dhd,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"metrics_json={args.output_json}")


if __name__ == "__main__":
    main()
