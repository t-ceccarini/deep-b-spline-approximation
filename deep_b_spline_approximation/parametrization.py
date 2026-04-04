# -*- coding: utf-8 -*-
"""Classical point parametrization baselines."""

import torch

torch.set_default_dtype(torch.float64)


def _as_batched_curves(curves):
    if curves.ndim != 3:
        raise ValueError(f"Expected curves with shape [batch, n_points, dim], got {tuple(curves.shape)}")
    return curves


def computeUniformParametrization(curves, device=None):
    curves = _as_batched_curves(curves)
    if device is None:
        device = curves.device
    n_points = curves.shape[1]
    t = torch.linspace(0.0, 1.0, n_points, dtype=curves.dtype, device=device)
    return t.unsqueeze(0).repeat(curves.shape[0], 1)


def _compute_distance_parametrization(curves, alpha, eps=1e-12):
    curves = _as_batched_curves(curves)
    segment_lengths = torch.norm(curves[:, 1:] - curves[:, :-1], dim=2).pow(alpha)
    cumulative = torch.cumsum(segment_lengths, dim=1)
    t = torch.cat((torch.zeros(curves.shape[0], 1, dtype=curves.dtype, device=curves.device), cumulative), dim=1)
    total = t[:, -1:].clamp_min(eps)
    return t / total


def computeChordLengthParametrization(curves):
    return _compute_distance_parametrization(curves, alpha=1.0)


def computeCentripetalParametrization(curves):
    return _compute_distance_parametrization(curves, alpha=0.5)
