# -*- coding: utf-8 -*-
"""Minimal reproducible PPN-MLP training entrypoint for P0."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_b_spline_approximation.ppn import (
    CentripetalResidualParametrizationNetwork,
    ConvexMixtureParametrizationNetwork,
    PointParametrizationNetwork,
)
from deep_b_spline_approximation.BSpline import computeBSplineApproximation
from deep_b_spline_approximation.parametrization import computeCentripetalParametrization
from deep_b_spline_approximation.parametrization_features import (
    build_enriched_curve_inputs,
    parse_parametrization_feature_names,
)
from deep_b_spline_approximation.reproducibility import set_global_seed
from deep_b_spline_approximation.synthetic_dataset import generate_synthetic_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPN-MLP model on a synthetic Bezier dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppn_mlp"))
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--min-internal-knots", type=int, default=None)
    parser.add_argument("--max-internal-knots", type=int, default=None)
    parser.add_argument("--fit-internal-knots", type=int, default=0)
    parser.add_argument("--non-uniform", action="store_true")
    parser.add_argument(
        "--model-kind",
        choices=["ppn_mlp", "baseline_mixture", "centripetal_residual"],
        default="ppn_mlp",
    )
    parser.add_argument("--normalization", choices=["batchnorm", "layernorm", "none"], default="batchnorm")
    parser.add_argument("--param-inputs", type=str, default="")
    parser.add_argument("--param-loss-weight", type=float, default=0.0)
    parser.add_argument("--centripetal-margin-weight", type=float, default=0.0)
    parser.add_argument("--centripetal-margin", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def compute_centripetal_margin_loss(target_batch, curve_loss, model, curve_criterion, margin_weight, margin):
    if margin_weight <= 0:
        return torch.tensor(0.0, dtype=curve_loss.dtype, device=curve_loss.device)
    baseline_params = computeCentripetalParametrization(target_batch)
    baseline_splines = computeBSplineApproximation(
        target_batch,
        baseline_params,
        device=target_batch.device,
        u=model.u.to(target_batch.device),
    )
    baseline_loss = curve_criterion(target_batch, baseline_splines)
    return torch.relu(curve_loss - baseline_loss + margin)


def evaluate(
    model,
    flat_curves,
    target_curves,
    target_params,
    curve_criterion,
    param_criterion,
    param_loss_weight,
    margin_weight,
    margin,
    batch_size,
    device,
):
    model.eval()
    curve_losses = []
    param_losses = []
    margin_losses = []
    with torch.no_grad():
        if target_params is None:
            loader = DataLoader(TensorDataset(flat_curves, target_curves), batch_size=batch_size, shuffle=False)
        else:
            loader = DataLoader(TensorDataset(flat_curves, target_curves, target_params), batch_size=batch_size, shuffle=False)
        for batch in loader:
            flat_batch = batch[0]
            target_batch = batch[1]
            flat_batch = flat_batch.to(device)
            target_batch = target_batch.to(device)
            splines, predicted_params, _ = model(flat_batch)
            curve_loss = curve_criterion(target_batch, splines)
            curve_losses.append(curve_loss.item())
            if target_params is not None and param_loss_weight > 0:
                target_param_batch = batch[2].to(device)
                param_losses.append(param_criterion(target_param_batch, predicted_params).item())
            if margin_weight > 0:
                margin_losses.append(
                    compute_centripetal_margin_loss(
                        target_batch,
                        curve_loss,
                        model,
                        curve_criterion,
                        margin_weight,
                        margin,
                    ).item()
                )
    mean_curve_loss = sum(curve_losses) / len(curve_losses)
    mean_param_loss = sum(param_losses) / len(param_losses) if param_losses else 0.0
    mean_margin_loss = sum(margin_losses) / len(margin_losses) if margin_losses else 0.0
    return (
        mean_curve_loss,
        mean_param_loss,
        mean_margin_loss,
        mean_curve_loss + param_loss_weight * mean_param_loss + margin_weight * mean_margin_loss,
    )


def train():
    args = parse_args()
    set_global_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesto ma non disponibile.")
    device = torch.device(args.device)

    use_param_supervision = args.param_loss_weight > 0
    train_dataset = generate_synthetic_dataset(
        n_curves=args.n_train,
        n_points=args.n_points,
        degree=3,
        internal_knots=None,
        min_internal_knots=args.min_internal_knots,
        max_internal_knots=args.max_internal_knots,
        non_uniform=args.non_uniform,
        seed=args.seed,
        return_params=use_param_supervision,
    )
    val_dataset = generate_synthetic_dataset(
        n_curves=args.n_val,
        n_points=args.n_points,
        degree=3,
        internal_knots=None,
        min_internal_knots=args.min_internal_knots,
        max_internal_knots=args.max_internal_knots,
        non_uniform=args.non_uniform,
        seed=args.seed + 1,
        return_params=use_param_supervision,
    )
    if use_param_supervision:
        _, train_curves, knots, train_params = train_dataset
        _, val_curves, _, val_params = val_dataset
        param_input_names = parse_parametrization_feature_names(args.param_inputs)
        train_flat = build_enriched_curve_inputs(train_curves, param_input_names)
        val_flat = build_enriched_curve_inputs(val_curves, param_input_names)
        loader = DataLoader(
            TensorDataset(train_flat, train_curves, train_params),
            batch_size=args.batch_size,
            shuffle=True,
        )
    else:
        _, train_curves, knots = train_dataset
        _, val_curves, _ = val_dataset
        param_input_names = parse_parametrization_feature_names(args.param_inputs)
        train_flat = build_enriched_curve_inputs(train_curves, param_input_names)
        val_flat = build_enriched_curve_inputs(val_curves, param_input_names)
        train_params = None
        val_params = None
        loader = DataLoader(TensorDataset(train_flat, train_curves), batch_size=args.batch_size, shuffle=True)

    fit_internal_knots = None
    if args.fit_internal_knots > 0:
        fit_internal_knots = np.linspace(0.0, 1.0, args.fit_internal_knots + 2, dtype=np.float64)[1:-1].tolist()

    if args.model_kind == "ppn_mlp":
        model = PointParametrizationNetwork(
            inputSize=args.n_points * (2 + len(param_input_names)),
            hiddenSize=args.hidden_size,
            p=2,
            k=3,
            device=device,
            internal_knots=fit_internal_knots,
            normalization=args.normalization,
            input_channels=2 + len(param_input_names),
        )
    else:
        if args.model_kind == "baseline_mixture":
            model = ConvexMixtureParametrizationNetwork(
                inputSize=args.n_points * 2,
                hiddenSize=args.hidden_size,
                p=2,
                k=3,
                device=device,
                internal_knots=fit_internal_knots,
                normalization=args.normalization,
            )
        else:
            model = CentripetalResidualParametrizationNetwork(
                inputSize=args.n_points * 2,
                hiddenSize=args.hidden_size,
                p=2,
                k=3,
                device=device,
                internal_knots=fit_internal_knots,
                normalization=args.normalization,
            )
    model.to(device)

    curve_criterion = nn.MSELoss()
    param_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_curve_loss = 0.0
        running_param_loss = 0.0
        running_margin_loss = 0.0
        running_total_loss = 0.0
        for batch in loader:
            flat_batch = batch[0].to(device)
            target_batch = batch[1].to(device)

            optimizer.zero_grad()
            splines, predicted_params, _ = model(flat_batch)
            curve_loss = curve_criterion(target_batch, splines)
            if use_param_supervision:
                target_param_batch = batch[2].to(device)
                param_loss = param_criterion(target_param_batch, predicted_params)
            else:
                param_loss = torch.tensor(0.0, dtype=curve_loss.dtype, device=device)
            margin_loss = compute_centripetal_margin_loss(
                target_batch=target_batch,
                curve_loss=curve_loss,
                model=model,
                curve_criterion=curve_criterion,
                margin_weight=args.centripetal_margin_weight,
                margin=args.centripetal_margin,
            )
            total_loss = (
                curve_loss
                + args.param_loss_weight * param_loss
                + args.centripetal_margin_weight * margin_loss
            )
            total_loss.backward()
            optimizer.step()

            running_curve_loss += curve_loss.item()
            running_param_loss += param_loss.item()
            running_margin_loss += margin_loss.item()
            running_total_loss += total_loss.item()

        train_curve_mse = running_curve_loss / len(loader)
        train_param_mse = running_param_loss / len(loader)
        train_margin_loss = running_margin_loss / len(loader)
        train_total_loss = running_total_loss / len(loader)
        val_curve_mse, val_param_mse, val_margin_loss, val_total_loss = evaluate(
            model=model,
            flat_curves=val_flat,
            target_curves=val_curves,
            target_params=val_params,
            curve_criterion=curve_criterion,
            param_criterion=param_criterion,
            param_loss_weight=args.param_loss_weight,
            margin_weight=args.centripetal_margin_weight,
            margin=args.centripetal_margin,
            batch_size=args.batch_size,
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_mse": train_curve_mse,
            "train_param_mse": train_param_mse,
            "train_margin_loss": train_margin_loss,
            "train_total_loss": train_total_loss,
            "val_mse": val_curve_mse,
            "val_param_mse": val_param_mse,
            "val_margin_loss": val_margin_loss,
            "val_total_loss": val_total_loss,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_mse={train_curve_mse:.8f} train_param_mse={train_param_mse:.8f} "
            f"train_margin={train_margin_loss:.8f} val_mse={val_curve_mse:.8f} "
            f"val_param_mse={val_param_mse:.8f} val_margin={val_margin_loss:.8f}"
        )

    checkpoint_path = args.output_dir / "ppn_mlp.pt"
    checkpoint_config = vars(args).copy()
    checkpoint_config["output_dir"] = str(args.output_dir)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": checkpoint_config,
            "knots": knots.tolist(),
        },
        checkpoint_path,
    )

    metrics_json = args.output_dir / "metrics.json"
    metrics_csv = args.output_dir / "metrics.csv"
    metrics_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_mse",
                "train_param_mse",
                "train_margin_loss",
                "train_total_loss",
                "val_mse",
                "val_param_mse",
                "val_margin_loss",
                "val_total_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"checkpoint={checkpoint_path}")
    print(f"metrics_json={metrics_json}")
    print(f"metrics_csv={metrics_csv}")


if __name__ == "__main__":
    train()
