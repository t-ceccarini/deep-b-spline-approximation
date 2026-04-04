# -*- coding: utf-8 -*-
"""Build enriched PPN inputs by appending classical parametrization channels."""

import torch

from .parametrization import (
    computeCentripetalParametrization,
    computeChordLengthParametrization,
    computeUniformParametrization,
)


_PARAMETRIZATION_BUILDERS = {
    "uniform": computeUniformParametrization,
    "chord_length": computeChordLengthParametrization,
    "centripetal": computeCentripetalParametrization,
}


def parse_parametrization_feature_names(raw_value):
    if raw_value is None or raw_value.strip() == "":
        return []
    names = [name.strip() for name in raw_value.split(",") if name.strip()]
    unsupported = [name for name in names if name not in _PARAMETRIZATION_BUILDERS]
    if unsupported:
        raise ValueError(f"Unsupported parametrization feature(s): {unsupported}")
    return names


def build_enriched_curve_inputs(curves, parametrization_feature_names):
    if not parametrization_feature_names:
        return curves.permute(0, 2, 1).reshape(curves.shape[0], -1)

    channels = [curves]
    for name in parametrization_feature_names:
        params = _PARAMETRIZATION_BUILDERS[name](curves).unsqueeze(-1)
        channels.append(params)

    enriched = torch.cat(channels, dim=2)
    return enriched.permute(0, 2, 1).reshape(curves.shape[0], -1)
