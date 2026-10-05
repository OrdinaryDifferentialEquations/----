#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import root
from scipy.stats import qmc

from model import (
    PRNNTrainer,
    ReducedPDE,
    TrainConfig,
    TrainResult,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

ROOT = Path(r"F:\科研相关\方程复现\Burgers Equation")
CODE_DIR = ROOT / "code"
RESULT_DIR = ROOT / "result"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
DATA_DIR = RESULT_DIR / "data"

N_X = 129
N_P = N_X - 1

MU_LOWER = np.asarray([1.0, 1.0], dtype=np.float64)
MU_UPPER = np.asarray([10.0, 10.0], dtype=np.float64)

SNAPSHOT_COUNTS = [10, 80, 320]
M_VALUES = [2,4,6,8]
OMEGA_VALUES = [1.0e-2, 1.0, 100.0]

N_RESIDUAL = 5000
N_VALIDATION = 40
TEST_AXIS_SIZE = 101

HIDDEN_WIDTH = 20
HIDDEN_DEPTH = 3

TOTAL_EPOCHS = 20000
SEED = 1234

NEWTON_MAX_ITER = 100
NEWTON_TOLERANCE = 1.0e-10

TORCH_DTYPE = torch.float32

COLORS = {
    "Projection": "#54200B",
    "POD-G": "#C23B22",
    "PDNN": "#D95F02",
    "PINN": "#D8A900",
    "PRNN": "#9E0142",
}

LINESTYLES = {
    "Projection": "-.",
    "POD-G": "-.",
    "PDNN": ":",
    "PINN": "--",
    "PRNN": "-",
}

MARKERS = {
    "Projection": "p",
    "POD-G": "H",
    "PDNN": "^",
    "PINN": "s",
    "PRNN": "D",
}

@dataclass
class PODData:
    basis_full: np.ndarray
    singular_values: np.ndarray
    snapshots: np.ndarray
    parameters: np.ndarray
    projection_full: np.ndarray

@dataclass
class BurgersROM:
    basis: np.ndarray
    tensor_a: np.ndarray
    matrix_b: np.ndarray
    x: np.ndarray
    d1: np.ndarray
    d2: np.ndarray

def configure_plotting() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "savefig.dpi": 400,
            "figure.dpi": 120,
        }
    )

def chebyshev_differentiation(
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if order == 0:
        return np.asarray([1.0]), np.asarray([[0.0]])

    x = np.cos(np.pi * np.arange(order + 1) / order)

    c = np.ones(order + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(order + 1))

    x_column = x[:, None]
    difference = x_column - x_column.T

    d1 = (
        np.outer(c, 1.0 / c)
        / (difference + np.eye(order + 1))
    )

    d1 = d1 - np.diag(np.sum(d1, axis=1))

    sort_indices = np.argsort(x)
    x = x[sort_indices]
    d1 = d1[np.ix_(sort_indices, sort_indices)]
    d2 = d1 @ d1

    return x, d1, d2

def exact_solution(
    x: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    parameters = np.asarray(parameters, dtype=np.float64)

    mu1 = parameters[:, 0:1]
    mu2 = parameters[:, 1:2]

    return (
        (1.0 + mu1 * x)
        * np.sin(-mu2 * x / 3.0)
        * (x ** 2 - 1.0)
    )

def source_term(
    x: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    parameters = np.asarray(parameters, dtype=np.float64)

    mu1 = parameters[:, 0:1]
    mu2 = parameters[:, 1:2]

    f1 = (1.0 + mu1 * x) * (x ** 2 - 1.0)
    f2 = np.sin(-mu2 * x / 3.0)

    f1_x = 3.0 * mu1 * x ** 2 + 2.0 * x - mu1
    f2_x = -(mu2 / 3.0) * np.cos(-mu2 * x / 3.0)

    f1_xx = 6.0 * mu1 * x + 2.0
    f2_xx = -(mu2 ** 2 / 9.0) * np.sin(-mu2 * x / 3.0)

    phi = f1 * f2
    phi_x = f1 * f2_x + f1_x * f2
    phi_xx = (
        2.0 * f1_x * f2_x
        + f1 * f2_xx
        + f1_xx * f2
    )

    source = phi * phi_x - phi_xx

    source[:, 0] = phi[:, 0]
    source[:, -1] = phi[:, -1]

    return source

def generate_parameter_sets() -> Dict[str, np.ndarray]:
    sampler = qmc.Sobol(
        d=2,
        scramble=False,
        seed=SEED,
    )

    unit_snapshot_parameters = sampler.random(320)
    snapshot_parameters = qmc.scale(
        unit_snapshot_parameters,
        MU_LOWER,
        MU_UPPER,
    )

    residual_sampler = qmc.LatinHypercube(
        d=2,
        seed=SEED + 1,
    )

    residual_parameters = qmc.scale(
        residual_sampler.random(N_RESIDUAL),
        MU_LOWER,
        MU_UPPER,
    )

    validation_rng = np.random.default_rng(SEED + 2)

    validation_parameters = validation_rng.uniform(
        MU_LOWER,
        MU_UPPER,
        size=(N_VALIDATION, 2),
    )

    test_axis_1 = np.linspace(
        MU_LOWER[0],
        MU_UPPER[0],
        TEST_AXIS_SIZE,
    )

    test_axis_2 = np.linspace(
        MU_LOWER[1],
        MU_UPPER[1],
        TEST_AXIS_SIZE,
    )

    grid_1, grid_2 = np.meshgrid(
        test_axis_1,
        test_axis_2,
        indexing="xy",
    )

    test_parameters = np.column_stack(
        [grid_1.ravel(), grid_2.ravel()]
    )

    return {
        "snapshots": snapshot_parameters,
        "residual": residual_parameters,
        "validation": validation_parameters,
        "test": test_parameters,
    }

def build_pod(
    x: np.ndarray,
    parameters: np.ndarray,
) -> PODData:
    snapshots_by_parameter = exact_solution(x, parameters)
    snapshot_matrix = snapshots_by_parameter.T

    basis_full, singular_values, _ = np.linalg.svd(
        snapshot_matrix,
        full_matrices=False,
    )

    projection_full = (
        snapshots_by_parameter @ basis_full
    )

    return PODData(
        basis_full=basis_full,
        singular_values=singular_values,
        snapshots=snapshot_matrix,
        parameters=parameters,
        projection_full=projection_full,
    )

def build_rom(
    basis: np.ndarray,
    x: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
) -> BurgersROM:
    m = basis.shape[1]
    basis_derivative = d1 @ basis

    tensor_a = np.einsum(
        "xk,xi,xj->kij",
        basis,
        basis,
        basis_derivative,
    )

    tensor_a[:, :, :] -= np.einsum(
        "kij->kij",
        np.zeros_like(tensor_a),
    )

    operator_b = -d2.copy()

    operator_b[0, :] = 0.0
    operator_b[-1, :] = 0.0
    operator_b[0, 0] = 1.0
    operator_b[-1, -1] = 1.0

    matrix_b = basis.T @ operator_b @ basis

    nonlinear_rows = np.einsum(
        "xi,xj->xij",
        basis,
        basis_derivative,
    )

    nonlinear_rows[0, :, :] = 0.0
    nonlinear_rows[-1, :, :] = 0.0

    tensor_a = np.einsum(
        "xk,xij->kij",
        basis,
        nonlinear_rows,
    )

    return BurgersROM(
        basis=basis,
        tensor_a=tensor_a,
        matrix_b=matrix_b,
        x=x,
        d1=d1,
        d2=d2,
    )

def projected_source(
    rom: BurgersROM,
    parameters: np.ndarray,
) -> np.ndarray:
    source = source_term(rom.x, parameters)
    return source @ rom.basis

def reduced_residual_numpy(
    rom: BurgersROM,
    coefficients: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)

    nonlinear = np.einsum(
        "bi,kij,bj->bk",
        coefficients,
        rom.tensor_a,
        coefficients,
    )

    linear = coefficients @ rom.matrix_b.T
    source = projected_source(rom, parameters)

    return nonlinear + linear - source

def reduced_jacobian_numpy(
    rom: BurgersROM,
    coefficient: np.ndarray,
) -> np.ndarray:
    coefficient = np.asarray(
        coefficient,
        dtype=np.float64,
    ).reshape(-1)

    nonlinear_jacobian = np.einsum(
        "kij,j->ki",
        rom.tensor_a,
        coefficient,
    )

    nonlinear_jacobian += np.einsum(
        "kji,j->ki",
        rom.tensor_a,
        coefficient,
    )

    return nonlinear_jacobian + rom.matrix_b

def make_torch_residual(
    rom: BurgersROM,
    device: torch.device,
):
    tensor_a = torch.as_tensor(
        rom.tensor_a,
        dtype=TORCH_DTYPE,
        device=device,
    )

    matrix_b = torch.as_tensor(
        rom.matrix_b,
        dtype=TORCH_DTYPE,
        device=device,
    )

    basis = torch.as_tensor(
        rom.basis,
        dtype=TORCH_DTYPE,
        device=device,
    )

    x = torch.as_tensor(
        rom.x[None, :],
        dtype=TORCH_DTYPE,
        device=device,
    )

    def residual(
        coefficients: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        mu1 = parameters[:, 0:1]
        mu2 = parameters[:, 1:2]

        f1 = (1.0 + mu1 * x) * (x ** 2 - 1.0)
        f2 = torch.sin(-mu2 * x / 3.0)

        f1_x = 3.0 * mu1 * x ** 2 + 2.0 * x - mu1
        f2_x = -(mu2 / 3.0) * torch.cos(
            -mu2 * x / 3.0
        )

        f1_xx = 6.0 * mu1 * x + 2.0
        f2_xx = -(mu2 ** 2 / 9.0) * torch.sin(
            -mu2 * x / 3.0
        )

        phi = f1 * f2
        phi_x = f1 * f2_x + f1_x * f2
        phi_xx = (
            2.0 * f1_x * f2_x
            + f1 * f2_xx
            + f1_xx * f2
        )

        source = phi * phi_x - phi_xx
        source = source.clone()
        source[:, 0] = phi[:, 0]
        source[:, -1] = phi[:, -1]
        source_reduced = source @ basis

        nonlinear = torch.einsum(
            "bi,kij,bj->bk",
            coefficients,
            tensor_a,
            coefficients,
        )

        linear = coefficients @ matrix_b.T

        return nonlinear + linear - source_reduced

    return residual

def relative_l2_errors(
    exact_fields: np.ndarray,
    predicted_fields: np.ndarray,
) -> np.ndarray:
    numerator = np.linalg.norm(
        exact_fields - predicted_fields,
        axis=1,
    )

    denominator = np.linalg.norm(
        exact_fields,
        axis=1,
    )

    denominator = np.maximum(denominator, 1.0e-14)
    return numerator / denominator

def mean_relative_l2_error(
    exact_fields: np.ndarray,
    predicted_fields: np.ndarray,
) -> float:
    return float(
        np.mean(
            relative_l2_errors(
                exact_fields,
                predicted_fields,
            )
        )
    )

def projection_error(
    pod: PODData,
    m: int,
    x: np.ndarray,
    parameters: np.ndarray,
) -> float:
    basis = pod.basis_full[:, :m]
    exact_fields = exact_solution(x, parameters)
    coefficients = exact_fields @ basis
    predicted_fields = coefficients @ basis.T

    return mean_relative_l2_error(
        exact_fields,
        predicted_fields,
    )

def pod_g_coefficients(
    rom: BurgersROM,
    pod: PODData,
    parameters: np.ndarray,
) -> np.ndarray:
    exact_training = pod.snapshots.T
    training_projection = exact_training @ rom.basis

    output = np.zeros(
        (len(parameters), rom.basis.shape[1]),
        dtype=np.float64,
    )

    for index, parameter in enumerate(parameters):
        distances = np.linalg.norm(
            pod.parameters - parameter[None, :],
            axis=1,
        )

        initial = training_projection[
            int(np.argmin(distances))
        ].copy()

        def function(value: np.ndarray) -> np.ndarray:
            return reduced_residual_numpy(
                rom,
                value[None, :],
                parameter[None, :],
            )[0]

        def jacobian(value: np.ndarray) -> np.ndarray:
            return reduced_jacobian_numpy(rom, value)

        result = root(
            function,
            initial,
            jac=jacobian,
            method="hybr",
            options={
                "xtol": NEWTON_TOLERANCE,
                "maxfev": NEWTON_MAX_ITER * (
                    rom.basis.shape[1] + 1
                ),
            },
        )

        if result.success and np.all(np.isfinite(result.x)):
            output[index] = result.x
            continue

        value = initial.copy()

        for _ in range(NEWTON_MAX_ITER):
            residual = function(value)

            if np.linalg.norm(residual) <= NEWTON_TOLERANCE:
                break

            jacobian_value = jacobian(value)

            try:
                increment = np.linalg.solve(
                    jacobian_value,
                    -residual,
                )
            except np.linalg.LinAlgError:
                increment = np.linalg.lstsq(
                    jacobian_value,
                    -residual,
                    rcond=None,
                )[0]

            value = value + increment

        output[index] = value

    return output

def pod_g_error(
    rom: BurgersROM,
    pod: PODData,
    parameters: np.ndarray,
    cache_path: Path,
) -> float:
    if cache_path.exists():
        coefficients = np.load(cache_path)["coefficients"]
    else:
        coefficients = pod_g_coefficients(
            rom,
            pod,
            parameters,
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            cache_path,
            coefficients=coefficients,
            parameters=parameters,
        )

    exact_fields = exact_solution(rom.x, parameters)
    predicted_fields = coefficients @ rom.basis.T

    return mean_relative_l2_error(
        exact_fields,
        predicted_fields,
    )

def network_error(
    trainer: PRNNTrainer,
    result: TrainResult,
    rom: BurgersROM,
    parameters: np.ndarray,
) -> float:
    coefficients = trainer.predict(
        result.model,
        result.scaling,
        parameters,
    )

    exact_fields = exact_solution(rom.x, parameters)
    predicted_fields = coefficients @ rom.basis.T

    return mean_relative_l2_error(
        exact_fields,
        predicted_fields,
    )

def checkpoint_path(
    method: str,
    n_snapshots: int,
    m: int,
    omega: Optional[float] = None,
) -> Path:
    if omega is None:
        suffix = ""
    else:
        suffix = f"_omega{omega:.0e}"

    return (
        CHECKPOINT_DIR
        / f"{method}_Ns{n_snapshots}_m{m}{suffix}.pt"
    )

def train_or_load(
    trainer: PRNNTrainer,
    method: str,
    projection_parameters: np.ndarray,
    projection_coefficients: np.ndarray,
    residual_parameters: np.ndarray,
    n_snapshots: int,
    m: int,
    omega: float,
    device: str,
    force: bool,
) -> TrainResult:
    path = checkpoint_path(
        method,
        n_snapshots,
        m,
        omega if method == "PRNN" else None,
    )

    if path.exists() and not force:
        return load_checkpoint(
            path,
            hidden_width=HIDDEN_WIDTH,
            hidden_depth=HIDDEN_DEPTH,
            device=device,
            dtype=TORCH_DTYPE,
        )

    config = TrainConfig(
        epochs=TOTAL_EPOCHS,
        adam_fraction=0.90,
        learning_rate=1.0e-2,
        learning_rate_gamma=0.96,
        learning_rate_step=200,
        residual_batches=10,
        projection_batches=1,
        pdnn_weight_decay=1.0e-4,
        physics_weight=omega,
        seed=SEED,
        lbfgs_lr=1.0,
        lbfgs_history_size=50,
        lbfgs_tolerance_grad=1.0e-10,
        lbfgs_tolerance_change=1.0e-12,
        log_every=200,
    )

    result = trainer.train(
        method=method,
        projection_parameters=projection_parameters,
        projection_coefficients=projection_coefficients,
        residual_parameters=residual_parameters,
        config=config,
    )

    save_checkpoint(
        path,
        result,
        metadata={
            "n_snapshots": n_snapshots,
            "m": m,
            "omega": omega,
            "hidden_width": HIDDEN_WIDTH,
            "hidden_depth": HIDDEN_DEPTH,
            "n_residual": N_RESIDUAL,
            "epochs": TOTAL_EPOCHS,
        },
    )

    return result

def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    path = RESULT_DIR / filename

    figure.savefig(
        path,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.0,
        facecolor="white",
    )

    plt.close(figure)

def plot_singular_values(
    singular_values: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 6.0))

    indices = np.arange(1, len(singular_values) + 1)

    axis.semilogy(
        indices,
        singular_values,
        color=COLORS["Projection"],
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.1,
    )

    axis.set_xlabel(r"$m$")
    axis.set_ylabel("Singular value")
    axis.set_xlim(1, len(singular_values))
    axis.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.6,
        alpha=0.28,
    )

    figure.tight_layout(pad=0.2)

    save_figure(
        figure,
        "Figure_1_Singular_Values_Ns80.png",
    )

def configuration_suffix() -> str:
    ns_text = "_".join(str(value) for value in SNAPSHOT_COUNTS)
    m_text = "_".join(str(value) for value in M_VALUES)
    return f"Ns{ns_text}_m{m_text}"

def plot_error_comparison(
    error_results: Dict[int, Dict[str, List[float]]],
) -> None:
    snapshot_counts = list(error_results.keys())
    number_of_panels = len(snapshot_counts)

    if number_of_panels == 0:
        raise ValueError("No error results are available for plotting.")

    if number_of_panels == 1:
        figure_size = (8.0, 6.0)
    else:
        figure_size = (
            4.0 * number_of_panels,
            3.0 * number_of_panels,
        )

    figure, axes = plt.subplots(
        1,
        number_of_panels,
        figsize=figure_size,
        sharey=True,
        squeeze=False,
    )

    axes = axes.ravel()

    methods = [
        "Projection",
        "POD-G",
        "PDNN",
        "PINN",
        "PRNN",
    ]

    for axis, n_snapshots in zip(
        axes,
        snapshot_counts,
    ):
        curves = error_results[n_snapshots]

        for method in methods:
            values = curves.get(method, [])

            if len(values) != len(M_VALUES):
                raise ValueError(
                    f"{method}, Ns={n_snapshots}: "
                    f"expected {len(M_VALUES)} values, "
                    f"but received {len(values)}."
                )

            axis.semilogy(
                M_VALUES,
                values,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker=MARKERS[method],
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
                label=method,
            )

        axis.set_xlabel(r"$m$")
        axis.set_title(rf"$N_s={n_snapshots}$")
        axis.set_xticks(M_VALUES)

        axis.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.55,
            alpha=0.25,
        )

    axes[0].set_ylabel(r"Error $\varepsilon$")

    axes[0].legend(
        loc="lower left",
        frameon=False,
        handlelength=2.5,
    )

    figure.tight_layout(
        pad=0.25,
        w_pad=0.55,
    )

    save_figure(
        figure,
        f"Figure_2_Error_Comparison_"
        f"{configuration_suffix()}.png",
    )

def plot_optimal_omega(
    omega_results: Dict[int, List[float]],
) -> None:
    if not omega_results:
        raise ValueError(
            "No quasi-optimal omega results are available."
        )

    figure, axis = plt.subplots(figsize=(8.0, 6.0))

    color_cycle = [
        "#D95F02",
        "#D8A900",
        "#9E0142",
        "#54200B",
        "#C23B22",
        "#6A3D9A",
    ]

    marker_cycle = [
        "^",
        "o",
        "s",
        "D",
        "p",
        "H",
    ]

    for index, n_snapshots in enumerate(
        omega_results.keys()
    ):
        values = omega_results[n_snapshots]

        if len(values) != len(M_VALUES):
            raise ValueError(
                f"Ns={n_snapshots}: expected "
                f"{len(M_VALUES)} omega values, "
                f"but received {len(values)}."
            )

        color = color_cycle[index % len(color_cycle)]
        marker = marker_cycle[index % len(marker_cycle)]

        axis.semilogy(
            M_VALUES,
            values,
            color=color,
            linestyle="-",
            marker=marker,
            markersize=6.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=rf"$N_s={n_snapshots}$",
        )

    axis.set_xlabel(r"$m$")
    axis.set_ylabel(r"Quasi-optimal weight $\omega$")
    axis.set_xticks(M_VALUES)

    positive_omega_values = sorted(
        {
            float(value)
            for value in OMEGA_VALUES
            if float(value) > 0.0
        }
    )

    if positive_omega_values:
        axis.set_yticks(positive_omega_values)

    axis.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.6,
        alpha=0.27,
    )

    axis.legend(
        loc="best",
        frameon=False,
    )

    figure.tight_layout(pad=0.2)

    save_figure(
        figure,
        f"Figure_3_Optimal_Omega_"
        f"{configuration_suffix()}.png",
    )

def plot_truth_prnn_heatmaps(
    x: np.ndarray,
    parameters: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    n_snapshots: int,
    m: int,
) -> None:
    number_of_samples = len(parameters)

    if number_of_samples == 0:
        raise ValueError(
            "At least one parameter is required "
            "for the heatmap comparison."
        )

    figure, axes = plt.subplots(
        2,
        number_of_samples,
        figsize=(8.0, 6.0),
        sharex=True,
        squeeze=False,
    )

    minimum = float(
        min(
            np.min(truth),
            np.min(prediction),
        )
    )

    maximum = float(
        max(
            np.max(truth),
            np.max(prediction),
        )
    )

    if minimum < 0.0 < maximum:
        norm = TwoSlopeNorm(
            vmin=minimum,
            vcenter=0.0,
            vmax=maximum,
        )
    else:
        norm = None

    image = None

    for column in range(number_of_samples):
        fields = [
            truth[column],
            prediction[column],
        ]

        for row, field in enumerate(fields):
            image = axes[row, column].imshow(
                field[None, :],
                extent=[x[0], x[-1], 0.0, 1.0],
                aspect="auto",
                origin="lower",
                cmap="RdYlBu_r",
                norm=norm,
                vmin=(
                    None
                    if norm is not None
                    else minimum
                ),
                vmax=(
                    None
                    if norm is not None
                    else maximum
                ),
            )

            axes[row, column].set_yticks([])

        mu1, mu2 = parameters[column]

        axes[0, column].set_title(
            rf"$\boldsymbol{{\mu}}="
            rf"({mu1:.2f},{mu2:.2f})$"
        )

        axes[1, column].set_xlabel(r"$x$")

    axes[0, 0].set_ylabel("Truth")
    axes[1, 0].set_ylabel("PRNN")

    color_axis = figure.add_axes(
        [0.91, 0.16, 0.022, 0.68]
    )

    figure.colorbar(
        image,
        cax=color_axis,
        label=r"$\phi(x;\boldsymbol{\mu})$",
    )

    figure.subplots_adjust(
        left=0.07,
        right=0.89,
        bottom=0.10,
        top=0.91,
        wspace=0.14,
        hspace=0.18,
    )

    save_figure(
        figure,
        f"Figure_4_Truth_PRNN_"
        f"Ns{n_snapshots}_m{m}.png",
    )

def write_error_csv(
    error_results: Dict[int, Dict[str, List[float]]],
    omega_results: Dict[int, List[float]],
) -> None:
    path = RESULT_DIR / "numerical_results.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Ns",
                "m",
                "Projection",
                "POD-G",
                "PDNN",
                "PINN",
                "PRNN",
                "optimal_omega",
            ]
        )

        for n_snapshots in SNAPSHOT_COUNTS:
            for index, m in enumerate(M_VALUES):
                writer.writerow(
                    [
                        n_snapshots,
                        m,
                        error_results[n_snapshots][
                            "Projection"
                        ][index],
                        error_results[n_snapshots][
                            "POD-G"
                        ][index],
                        error_results[n_snapshots][
                            "PDNN"
                        ][index],
                        error_results[n_snapshots][
                            "PINN"
                        ][index],
                        error_results[n_snapshots][
                            "PRNN"
                        ][index],
                        omega_results[n_snapshots][index],
                    ]
                )

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce steady Burgers PRNN experiment."
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain networks and overwrite checkpoints.",
    )

    args = parser.parse_args()

    configure_plotting()
    set_seed(SEED)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    x, d1, d2 = chebyshev_differentiation(N_P)
    parameter_sets = generate_parameter_sets()

    np.savez_compressed(
        DATA_DIR / "parameter_sets.npz",
        **parameter_sets,
    )

    pod_by_sample_count: Dict[int, PODData] = {}

    for n_snapshots in SNAPSHOT_COUNTS:
        parameters = parameter_sets["snapshots"][
            :n_snapshots
        ]

        pod_by_sample_count[n_snapshots] = build_pod(
            x,
            parameters,
        )

    plot_singular_values(
        pod_by_sample_count[80].singular_values
    )

    test_parameters = parameter_sets["test"]
    validation_parameters = parameter_sets["validation"]
    residual_parameters = parameter_sets["residual"]

    error_results: Dict[
        int,
        Dict[str, List[float]],
    ] = {}

    omega_results: Dict[int, List[float]] = {}
    selected_prnn: Dict[Tuple[int, int], TrainResult] = {}
    selected_trainers: Dict[Tuple[int, int], PRNNTrainer] = {}
    selected_roms: Dict[Tuple[int, int], BurgersROM] = {}

    for n_snapshots in SNAPSHOT_COUNTS:
        pod = pod_by_sample_count[n_snapshots]

        error_results[n_snapshots] = {
            "Projection": [],
            "POD-G": [],
            "PDNN": [],
            "PINN": [],
            "PRNN": [],
        }

        omega_results[n_snapshots] = []

        print("\n" + "=" * 78)
        print(f"Ns = {n_snapshots}")
        print("=" * 78)

        for m in M_VALUES:
            print(f"\nNs={n_snapshots}, m={m}")

            basis = pod.basis_full[:, :m]
            projection_coefficients = (
                pod.snapshots.T @ basis
            )

            rom = build_rom(
                basis,
                x,
                d1,
                d2,
            )

            device_object = torch.device(
                "cuda"
                if (
                    args.device == "cuda"
                    or (
                        args.device == "auto"
                        and torch.cuda.is_available()
                    )
                )
                else "cpu"
            )

            residual_function = make_torch_residual(
                rom,
                device_object,
            )

            reduced_pde = ReducedPDE(
                basis=basis,
                input_lower=MU_LOWER,
                input_upper=MU_UPPER,
                residual_function=residual_function,
                reference=np.zeros(N_X),
            )

            trainer = PRNNTrainer(
                reduced_pde=reduced_pde,
                hidden_width=HIDDEN_WIDTH,
                hidden_depth=HIDDEN_DEPTH,
                device=args.device,
                dtype=TORCH_DTYPE,
            )

            projection_value = projection_error(
                pod,
                m,
                x,
                test_parameters,
            )

            pod_g_value = pod_g_error(
                rom,
                pod,
                test_parameters,
                DATA_DIR
                / f"PODG_Ns{n_snapshots}_m{m}.npz",
            )

            pdnn_result = train_or_load(
                trainer=trainer,
                method="PDNN",
                projection_parameters=pod.parameters,
                projection_coefficients=(
                    projection_coefficients
                ),
                residual_parameters=residual_parameters,
                n_snapshots=n_snapshots,
                m=m,
                omega=1.0,
                device=args.device,
                force=args.force,
            )

            pdnn_value = network_error(
                trainer,
                pdnn_result,
                rom,
                test_parameters,
            )

            pinn_result = train_or_load(
                trainer=trainer,
                method="PINN",
                projection_parameters=pod.parameters,
                projection_coefficients=(
                    projection_coefficients
                ),
                residual_parameters=residual_parameters,
                n_snapshots=n_snapshots,
                m=m,
                omega=1.0,
                device=args.device,
                force=args.force,
            )

            pinn_value = network_error(
                trainer,
                pinn_result,
                rom,
                test_parameters,
            )

            best_prnn_result: Optional[TrainResult] = None
            best_validation_error = np.inf
            best_omega = OMEGA_VALUES[0]

            for omega in OMEGA_VALUES:
                candidate = train_or_load(
                    trainer=trainer,
                    method="PRNN",
                    projection_parameters=pod.parameters,
                    projection_coefficients=(
                        projection_coefficients
                    ),
                    residual_parameters=residual_parameters,
                    n_snapshots=n_snapshots,
                    m=m,
                    omega=omega,
                    device=args.device,
                    force=args.force,
                )

                candidate_validation_error = network_error(
                    trainer,
                    candidate,
                    rom,
                    validation_parameters,
                )

                print(
                    f"  omega={omega:.2e}, "
                    f"validation error="
                    f"{candidate_validation_error:.6e}"
                )

                if (
                    candidate_validation_error
                    < best_validation_error
                ):
                    best_validation_error = (
                        candidate_validation_error
                    )
                    best_prnn_result = candidate
                    best_omega = omega

            if best_prnn_result is None:
                raise RuntimeError(
                    "No PRNN candidate was successfully trained."
                )

            best_prnn_result.best_validation_error = (
                best_validation_error
            )

            prnn_value = network_error(
                trainer,
                best_prnn_result,
                rom,
                test_parameters,
            )

            error_results[n_snapshots][
                "Projection"
            ].append(projection_value)

            error_results[n_snapshots][
                "POD-G"
            ].append(pod_g_value)

            error_results[n_snapshots][
                "PDNN"
            ].append(pdnn_value)

            error_results[n_snapshots][
                "PINN"
            ].append(pinn_value)

            error_results[n_snapshots][
                "PRNN"
            ].append(prnn_value)

            omega_results[n_snapshots].append(best_omega)

            selected_prnn[
                (n_snapshots, m)
            ] = best_prnn_result

            selected_trainers[
                (n_snapshots, m)
            ] = trainer

            selected_roms[
                (n_snapshots, m)
            ] = rom

            print(
                "  "
                f"Projection={projection_value:.6e}, "
                f"POD-G={pod_g_value:.6e}, "
                f"PDNN={pdnn_value:.6e}, "
                f"PINN={pinn_value:.6e}, "
                f"PRNN={prnn_value:.6e}, "
                f"omega={best_omega:.2e}"
            )

    write_error_csv(
        error_results,
        omega_results,
    )

    plot_error_comparison(error_results)
    plot_optimal_omega(omega_results)

    available_heatmap_keys = sorted(
        set(selected_prnn.keys())
        & set(selected_trainers.keys())
        & set(selected_roms.keys())
    )

    if not available_heatmap_keys:
        raise RuntimeError(
            "No trained PRNN model is available for "
            "the truth/PRNN comparison."
        )

    if 80 in SNAPSHOT_COUNTS:
        heatmap_snapshot_count = 80
    else:
        heatmap_snapshot_count = max(SNAPSHOT_COUNTS)

    available_m_for_heatmap = sorted(
        m
        for n_snapshots, m in available_heatmap_keys
        if n_snapshots == heatmap_snapshot_count
    )

    if not available_m_for_heatmap:
        heatmap_snapshot_count = max(
            n_snapshots
            for n_snapshots, _ in available_heatmap_keys
        )

        available_m_for_heatmap = sorted(
            m
            for n_snapshots, m in available_heatmap_keys
            if n_snapshots == heatmap_snapshot_count
        )

    heatmap_m = max(available_m_for_heatmap)
    heatmap_key = (
        heatmap_snapshot_count,
        heatmap_m,
    )

    print(
        "\nHeatmap model: "
        f"Ns={heatmap_snapshot_count}, m={heatmap_m}"
    )

    heatmap_rng = np.random.default_rng(SEED + 3)

    number_of_heatmap_samples = min(
        3,
        len(test_parameters),
    )

    heatmap_indices = heatmap_rng.choice(
        len(test_parameters),
        size=number_of_heatmap_samples,
        replace=False,
    )

    heatmap_parameters = test_parameters[
        heatmap_indices
    ]

    heatmap_result = selected_prnn[heatmap_key]
    heatmap_trainer = selected_trainers[heatmap_key]
    heatmap_rom = selected_roms[heatmap_key]

    heatmap_coefficients = heatmap_trainer.predict(
        heatmap_result.model,
        heatmap_result.scaling,
        heatmap_parameters,
    )

    heatmap_truth = exact_solution(
        x,
        heatmap_parameters,
    )

    heatmap_prediction = (
        heatmap_coefficients
        @ heatmap_rom.basis.T
    )

    plot_truth_prnn_heatmaps(
        x=x,
        parameters=heatmap_parameters,
        truth=heatmap_truth,
        prediction=heatmap_prediction,
        n_snapshots=heatmap_snapshot_count,
        m=heatmap_m,
    )

    elapsed = time.perf_counter() - start_time

    metadata = {
        "spatial_points": N_X,
        "snapshot_counts": SNAPSHOT_COUNTS,
        "m_values": M_VALUES,
        "omega_values": OMEGA_VALUES,
        "n_residual": N_RESIDUAL,
        "n_validation": N_VALIDATION,
        "test_axis_size": TEST_AXIS_SIZE,
        "test_size": len(test_parameters),
        "hidden_width": HIDDEN_WIDTH,
        "hidden_depth": HIDDEN_DEPTH,
        "epochs": TOTAL_EPOCHS,
        "seed": SEED,
        "newton_max_iter": NEWTON_MAX_ITER,
        "newton_tolerance": NEWTON_TOLERANCE,
        "heatmap_parameters": heatmap_parameters.tolist(),
        "optimal_omega": {
            str(key): value
            for key, value in omega_results.items()
        },
        "errors": {
            str(key): value
            for key, value in error_results.items()
        },
        "elapsed_seconds": elapsed,
        "notes": [
            "Errors use equation (34): mean pointwise relative L2 error.",
            "Snapshot points use an unscrambled nested Sobol sequence.",
            "PRNN omega is selected with 40 random validation parameters.",
            "The heatmap comparison uses Ns=80 and m=9.",
        ],
    }

    with (
        RESULT_DIR / "experiment_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 78)
    print("Completed")
    print(f"Result directory: {RESULT_DIR}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print("=" * 78)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
