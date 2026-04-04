# -*- coding: utf-8 -*-
"""Synthetic B-spline/Bezier dataset generation following the thesis setup."""

import numpy as np
import torch
from scipy.interpolate import BSpline

from .loadfromtxt import loadFlatDataset

torch.set_default_dtype(torch.float64)


def _build_open_knot_vector(degree, internal_knots=None):
    internal_knots = [] if internal_knots is None else list(internal_knots)
    return np.array([0.0] * (degree + 1) + internal_knots + [1.0] * (degree + 1), dtype=np.float64)


def _sample_internal_knots(min_internal_knots, max_internal_knots, rng):
    if min_internal_knots is None and max_internal_knots is None:
        return None

    min_knots = 0 if min_internal_knots is None else min_internal_knots
    max_knots = min_knots if max_internal_knots is None else max_internal_knots
    if min_knots < 0 or max_knots < min_knots:
        raise ValueError(
            f"Invalid internal knot range: min_internal_knots={min_internal_knots}, "
            f"max_internal_knots={max_internal_knots}"
        )
    n_internal_knots = int(rng.integers(min_knots, max_knots + 1))
    if n_internal_knots == 0:
        return None
    return np.sort(rng.uniform(0.0, 1.0, size=n_internal_knots)).tolist()


def _sample_parameter_values(n_points, rng, non_uniform=False):
    if not non_uniform:
        return np.linspace(0.0, 1.0, n_points, dtype=np.float64)

    interior = np.sort(rng.uniform(0.0, 1.0, size=n_points - 2))
    return np.concatenate(([0.0], interior, [1.0]))


def generate_control_points(n_control_points, mu=(10.0, 10.0), delta_mu=(1.0, 0.0), sigma=(1.0, 2.0), rng=None):
    rng = np.random.default_rng() if rng is None else rng
    center = np.array(mu, dtype=np.float64)
    step = np.array(delta_mu, dtype=np.float64)
    std = np.array(sigma, dtype=np.float64)

    control_points = np.zeros((n_control_points, 2), dtype=np.float64)
    for index in range(n_control_points):
        control_points[index] = rng.normal(loc=center, scale=std)
        center = center + step
    return control_points


def generate_synthetic_dataset(
    n_curves,
    n_points=100,
    degree=3,
    internal_knots=None,
    min_internal_knots=None,
    max_internal_knots=None,
    mu=(10.0, 10.0),
    delta_mu=(1.0, 0.0),
    sigma=(1.0, 2.0),
    non_uniform=False,
    point_noise_std=0.0,
    seed=52,
    return_params=False,
):
    rng = np.random.default_rng(seed)

    curves = np.zeros((n_curves, n_points, 2), dtype=np.float64)
    params = np.zeros((n_curves, n_points), dtype=np.float64)
    knots = None
    for curve_index in range(n_curves):
        sampled_internal_knots = internal_knots
        if internal_knots is None and (min_internal_knots is not None or max_internal_knots is not None):
            sampled_internal_knots = _sample_internal_knots(min_internal_knots, max_internal_knots, rng)

        knots = _build_open_knot_vector(degree=degree, internal_knots=sampled_internal_knots)
        n_control_points = len(knots) - degree - 1

        control_points = generate_control_points(
            n_control_points=n_control_points,
            mu=mu,
            delta_mu=delta_mu,
            sigma=sigma,
            rng=rng,
        )
        spline = BSpline(knots, control_points, degree)
        t = _sample_parameter_values(n_points=n_points, rng=rng, non_uniform=non_uniform)
        curves[curve_index] = spline(t)
        params[curve_index] = t

    if point_noise_std > 0:
        curves = curves + rng.normal(loc=0.0, scale=point_noise_std, size=curves.shape)

    flat_curves = torch.tensor(curves.transpose(0, 2, 1).reshape(n_curves, -1), dtype=torch.float64)
    curve_tensors = torch.tensor(curves, dtype=torch.float64)
    param_tensors = torch.tensor(params, dtype=torch.float64)
    if return_params:
        return flat_curves, curve_tensors, torch.tensor(knots, dtype=torch.float64), param_tensors
    return flat_curves, curve_tensors, torch.tensor(knots, dtype=torch.float64)


def load_parnet_txt_dataset(path, p=2):
    flatdataset, dataset = loadFlatDataset(path, p=p)
    n_curves = flatdataset.shape[0]
    n_points = dataset.shape[1]
    curves = torch.tensor(dataset.reshape(n_curves, p, n_points).transpose(0, 2, 1), dtype=torch.float64)
    flat_curves = torch.tensor(flatdataset, dtype=torch.float64)
    return flat_curves, curves
