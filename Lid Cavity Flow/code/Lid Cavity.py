#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.optimize import root
from scipy.stats import qmc

from model import (
    PODData,
    SteadyPRNNTrainer,
    SteadyReducedProblem,
    TrainConfig,
    TrainResult,
    build_pod,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

# ============================================================
# 0. Paths and experiment configuration
# ============================================================

ROOT = Path(r"F:\科研相关\方程复现\Lid Cavity Flow")

CODE_DIR = ROOT / "code"

RESULT_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"

POD_FILE_NAME = "LidDrivenPOD.mat"
VALIDATION_FILE_NAME = "LidDrivenValidation.mat"

SNAPSHOT_COUNTS = [30, 60, 100]
M_VALUES = [2, 4, 6, 8]
OMEGA_VALUES = [1.0e-2, 1.0, 100.0]

SINGULAR_VALUE_SNAPSHOT_COUNT = 100

N_RESIDUAL = 2048
N_VALIDATION_SELECTION = 50

PARAMETER_LOWER = np.asarray(
    [100.0, 60.0],
    dtype=np.float64,
)

PARAMETER_UPPER = np.asarray(
    [500.0, 120.0],
    dtype=np.float64,
)

# Cavity network: paper Section 4.1.2.
HIDDEN_WIDTH = 30
HIDDEN_DEPTH = 5
ACTIVATION = "silu"

# Requested reduced training schedule.
TOTAL_EPOCHS = 6000
LBFGS_EPOCHS = 200

LEARNING_RATE = 1.0e-2
LEARNING_RATE_GAMMA = 0.95
LEARNING_RATE_STEP = 200

PROJECTION_BATCHES = 1
RESIDUAL_BATCHES = 10

PDNN_L2_LAMBDA = 1.0e-10

SEED = 1234
TORCH_DTYPE = torch.float32

POD_G_TOLERANCE = 1.0e-6
POD_G_MAX_ITERATIONS = 100

N_VARIABLES = 3
N_LOADED_VARIABLES = 6

# Figure 4 uses three original validation indices if available.
PREFERRED_FIELD_INDICES = [23, 61, 119]

METHODS = [
    "Projection",
    "POD-G",
    "PDNN",
    "PINN",
    "PRNN",
]

METHOD_COLORS = {
    "Projection": "#222222",
    "POD-G": "#D73027",
    "PDNN": "#C9B900",
    "PINN": "#168A24",
    "PRNN": "#006D2C",
}

METHOD_LINESTYLES = {
    "Projection": "-.",
    "POD-G": "-.",
    "PDNN": ":",
    "PINN": "--",
    "PRNN": "-",
}

SNAPSHOT_MARKERS = [
    "^",
    "o",
    "s",
    "D",
    "p",
    "H",
]

# ============================================================
# 1. General utilities
# ============================================================

def configure_plotting() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.7,
            "savefig.dpi": 400,
            "figure.dpi": 120,
        }
    )

def find_file_recursively(
    root: Path,
    filename: str,
) -> Path:
    candidates = [
        root / filename,
        root / "data" / filename,
        root / "NumSols" / filename,
        root / "code" / filename,
        root / "code" / "data" / filename,
        root / "code" / "NumSols" / filename,
        (
            root
            / "NumSols"
            / "100_500and60_120"
            / filename
        ),
        (
            root
            / "code"
            / "NumSols"
            / "100_500and60_120"
            / filename
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    recursive_matches = list(
        root.rglob(filename)
    )

    if recursive_matches:
        return recursive_matches[0]

    checked = "\n".join(
        str(path)
        for path in candidates
    )

    raise FileNotFoundError(
        f"Could not find {filename}.\n"
        f"Checked standard paths:\n{checked}\n\n"
        f"Place the original open-source MAT file "
        f"under {root}."
    )

def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    path = FIGURE_DIR / filename

    figure.savefig(
        path,
        dpi=400,
        facecolor="white",
        pad_inches=0.0,
    )

    plt.close(figure)

def configuration_suffix() -> str:
    ns_text = "_".join(
        str(value)
        for value in SNAPSHOT_COUNTS
    )

    m_text = "_".join(
        str(value)
        for value in M_VALUES
    )

    return f"Ns{ns_text}_m{m_text}"

def checkpoint_path(
    method: str,
    n_snapshots: int,
    m: int,
    omega: Optional[float] = None,
) -> Path:
    suffix = ""

    if omega is not None:
        suffix = f"_omega{omega:.0e}"

    return (
        CHECKPOINT_DIR
        / (
            f"{method}_Ns{n_snapshots}_"
            f"m{m}{suffix}.pt"
        )
    )

# ============================================================
# 2. Chebyshev differentiation
# ============================================================

class Chebyshev1D:
    def __init__(
        self,
        left: float = -1.0,
        right: float = 1.0,
        order: int = 5,
    ) -> None:
        self.left = left
        self.right = right
        self.length = right - left
        self.order = order

    def differentiation_matrix(
        self,
        derivative_order: int = 1,
    ) -> np.ndarray:
        order = self.order

        matrix = np.zeros(
            (order + 1, order + 1),
            dtype=np.float64,
        )

        coefficient = np.ones(
            order + 1,
            dtype=np.float64,
        )

        coefficient[0] = 2.0
        coefficient[-1] = 2.0

        for i in range(order + 1):
            for j in range(order + 1):
                if i == j:
                    continue

                denominator = (
                    2.0
                    * np.sin(
                        (i + j)
                        * np.pi
                        / (2.0 * order)
                    )
                    * np.sin(
                        (j - i)
                        * np.pi
                        / (2.0 * order)
                    )
                )

                matrix[i, j] = (
                    coefficient[i]
                    / coefficient[j]
                    * (-1.0) ** (i + j)
                    / denominator
                )

            matrix[i, i] = -np.sum(
                matrix[i, :]
            )

        matrix[0, 0] = (
            2.0 * order * order + 1.0
        ) / 6.0

        matrix[-1, -1] = -matrix[0, 0]

        matrix = matrix[::-1, ::-1]
        matrix *= 2.0 / self.length

        output = matrix.copy()

        for _ in range(
            1,
            derivative_order,
        ):
            output = output @ matrix

        return output

    def pressure_differentiation_matrix(
        self,
    ) -> np.ndarray:
        """
        Open-source PN-PN-2 pressure derivative.
        """
        n = self.order

        nodes = -np.cos(
            np.arange(n + 1)
            * np.pi
            / n
        )

        matrix = np.zeros(
            (n + 1, n + 1),
            dtype=np.float64,
        )

        for j in range(1, n):
            for i in range(1, n):
                if i == j:
                    matrix[j, j] = (
                        1.5
                        * nodes[j]
                        / (
                            1.0
                            - nodes[j] ** 2
                        )
                    )
                else:
                    matrix[j, i] = (
                        (-1.0) ** (i + j)
                        * (
                            1.0
                            - nodes[i] ** 2
                        )
                        / (
                            (
                                1.0
                                - nodes[j] ** 2
                            )
                            * (
                                nodes[j]
                                - nodes[i]
                            )
                        )
                    )

        return matrix

    def grid(self) -> np.ndarray:
        return (
            0.5 * (self.left + self.right)
            - np.cos(
                np.arange(
                    self.order + 1
                )
                * np.pi
                / self.order
            )
            * 0.5
            * self.length
        )

class Chebyshev2D:
    def __init__(
        self,
        nx: int,
        ny: int,
    ) -> None:
        self.x_chebyshev = Chebyshev1D(
            -1.0,
            1.0,
            nx - 1,
        )

        self.y_chebyshev = Chebyshev1D(
            -1.0,
            1.0,
            ny - 1,
        )

    def first_derivatives(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.x_chebyshev
            .differentiation_matrix(1),
            self.y_chebyshev
            .differentiation_matrix(1),
        )

    def second_derivatives(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.x_chebyshev
            .differentiation_matrix(2),
            self.y_chebyshev
            .differentiation_matrix(2),
        )

    def pressure_derivatives(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.x_chebyshev
            .pressure_differentiation_matrix(),
            self.y_chebyshev
            .pressure_differentiation_matrix(),
        )

    def grid(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_grid, x_grid = np.meshgrid(
            self.y_chebyshev.grid(),
            self.x_chebyshev.grid(),
        )

        return x_grid, y_grid

# ============================================================
# 3. Original cavity data
# ============================================================

@dataclass
class CavityData:
    samples: np.ndarray
    parameters: np.ndarray
    design_space: np.ndarray
    field_shape: Tuple[int, int]

    validation_samples: np.ndarray
    validation_parameters: np.ndarray

def load_cavity_data(
    pod_path: Path,
    validation_path: Path,
) -> CavityData:
    pod_data = loadmat(pod_path)
    validation_data = loadmat(
        validation_path
    )

    required_pod_keys = {
        "Samples",
        "parameters",
        "design_space",
        "FieldShape",
    }

    missing = (
        required_pod_keys
        - set(pod_data.keys())
    )

    if missing:
        raise KeyError(
            f"{pod_path} is missing keys: "
            f"{sorted(missing)}"
        )

    required_validation_keys = {
        "Samples",
        "parameters",
    }

    missing_validation = (
        required_validation_keys
        - set(validation_data.keys())
    )

    if missing_validation:
        raise KeyError(
            f"{validation_path} is missing keys: "
            f"{sorted(missing_validation)}"
        )

    field_shape_array = np.asarray(
        pod_data["FieldShape"]
    ).reshape(-1)

    field_shape = (
        int(field_shape_array[0]),
        int(field_shape_array[1]),
    )

    return CavityData(
        samples=np.asarray(
            pod_data["Samples"],
            dtype=np.float64,
        ),
        parameters=np.asarray(
            pod_data["parameters"],
            dtype=np.float64,
        ),
        design_space=np.asarray(
            pod_data["design_space"],
            dtype=np.float64,
        ),
        field_shape=field_shape,
        validation_samples=np.asarray(
            validation_data["Samples"],
            dtype=np.float64,
        ),
        validation_parameters=np.asarray(
            validation_data["parameters"],
            dtype=np.float64,
        ),
    )

def extract_interior_snapshots(
    samples: np.ndarray,
    field_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reproduce the original open-source ordering:

        [p, u, v, dummy, omega, psi]

    Only interior p, u, v are retained.

    Return shape:
        [full_reduced_state_dimension, n_samples].
    """
    n_samples = samples.shape[1]

    reshaped = np.reshape(
        samples,
        (
            field_shape[0],
            field_shape[1],
            N_LOADED_VARIABLES,
            n_samples,
        ),
    )

    interior = reshaped[
        1:-1,
        1:-1,
        0:N_VARIABLES,
        :,
    ]

    return np.reshape(
        interior,
        (-1, n_samples),
    )

# ============================================================
# 4. Cavity reduced-order system
# ============================================================

class CavityReducedSystem:
    def __init__(
        self,
        cavity_data: CavityData,
        pod: PODData,
        m: int,
    ) -> None:
        self.cavity_data = cavity_data
        self.pod = pod
        self.m = m

        self.basis = pod.basis(m)
        self.field_shape = cavity_data.field_shape

        self.interior_shape = (
            self.field_shape[0] - 2,
            self.field_shape[1] - 2,
        )

        self.chebyshev = Chebyshev2D(
            self.field_shape[0],
            self.field_shape[1],
        )

        self.dx, self.dy = (
            self.chebyshev.first_derivatives()
        )

        self.d2x, self.d2y = (
            self.chebyshev.second_derivatives()
        )

        self.dxp, self.dyp = (
            self.chebyshev.pressure_derivatives()
        )

        self.interior_mask = np.zeros(
            self.field_shape,
            dtype=np.float64,
        )

        self.interior_mask[
            1:-1,
            1:-1,
        ] = 1.0

        self.boundary_mask = (
            1.0 - self.interior_mask
        )

        # Reproduce the original boundary state from
        # the first high-fidelity sample.
        first_sample = cavity_data.samples[:, 0]

        u_full = np.reshape(
            first_sample[
                1::N_LOADED_VARIABLES
            ],
            self.field_shape,
        )

        self.u_boundary = (
            u_full
            * self.boundary_mask
        )

        self.b_equations, self.b_boundary = (
            self._build_linear_terms()
        )

        self.a_equations, self.a_boundary = (
            self._build_quadratic_terms()
        )

    def vector_to_fields(
        self,
        vector: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_field = np.zeros(
            self.field_shape,
            dtype=np.float64,
        )

        u_field = np.zeros_like(p_field)
        v_field = np.zeros_like(p_field)

        p_field[
            1:-1,
            1:-1,
        ] = np.reshape(
            vector[0::N_VARIABLES],
            self.interior_shape,
        )

        u_field[
            1:-1,
            1:-1,
        ] = np.reshape(
            vector[1::N_VARIABLES],
            self.interior_shape,
        )

        v_field[
            1:-1,
            1:-1,
        ] = np.reshape(
            vector[2::N_VARIABLES],
            self.interior_shape,
        )

        return (
            p_field,
            u_field,
            v_field,
        )

    def derivative_x(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        return self.dx @ field

    def derivative_y(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        return (
            self.dy @ field.T
        ).T

    def pressure_derivative_x(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        return self.dxp @ field

    def pressure_derivative_y(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        return (
            self.dyp @ field.T
        ).T

    def second_derivative_x(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        return self.derivative_x(
            self.derivative_x(field)
        )

    def second_derivative_y(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        return self.derivative_y(
            self.derivative_y(field)
        )

    def mixed_derivative(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        return self.derivative_y(
            self.derivative_x(field)
        )

    def first_derivatives(
        self,
        field: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.derivative_x(field),
            self.derivative_y(field),
        )

    def second_derivatives(
        self,
        field: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        return (
            self.second_derivative_x(field),
            self.second_derivative_y(field),
            self.mixed_derivative(field),
        )

    def _build_quadratic_terms(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reproduce original LidDriven.py getA().
        """
        a_equations = np.zeros(
            (4, self.m, self.m, self.m),
            dtype=np.float64,
        )

        a_boundary = np.zeros(
            (4, self.m, self.m),
            dtype=np.float64,
        )

        u_boundary_x, u_boundary_y = (
            self.first_derivatives(
                self.u_boundary
            )
        )

        mode_fields = [
            self.vector_to_fields(
                self.basis[:, index]
            )
            for index in range(self.m)
        ]

        mode_derivatives = []

        for _, u_mode, v_mode in mode_fields:
            u_x, u_y = self.first_derivatives(
                u_mode
            )

            v_x, v_y = self.first_derivatives(
                v_mode
            )

            mode_derivatives.append(
                (u_x, u_y, v_x, v_y)
            )

        for j in range(self.m):
            (
                u_j_x,
                u_j_y,
                v_j_x,
                v_j_y,
            ) = mode_derivatives[j]

            for k in range(self.m):
                _, u_k, v_k = mode_fields[k]

                for i in range(self.m):
                    _, u_i, v_i = mode_fields[i]

                    a_equations[
                        0, k, i, j
                    ] = np.sum(
                        self.interior_mask
                        * (
                            u_i * u_j_x * u_k
                            + u_i * v_j_x * v_k
                        )
                    )

                    a_equations[
                        1, k, i, j
                    ] = np.sum(
                        self.interior_mask
                        * (
                            u_i * u_j_y * u_k
                            + u_i * v_j_y * v_k
                        )
                    )

                    a_equations[
                        2, k, i, j
                    ] = np.sum(
                        self.interior_mask
                        * (
                            v_i * u_j_x * u_k
                            + v_i * v_j_x * v_k
                        )
                    )

                    a_equations[
                        3, k, i, j
                    ] = np.sum(
                        self.interior_mask
                        * (
                            v_i * u_j_y * u_k
                            + v_i * v_j_y * v_k
                        )
                    )

                    a_boundary[
                        0, k, i
                    ] = np.sum(
                        self.interior_mask
                        * (
                            u_i
                            * u_boundary_x
                            * u_k
                        )
                    )

                    a_boundary[
                        1, k, i
                    ] = np.sum(
                        self.interior_mask
                        * (
                            u_i
                            * u_boundary_y
                            * u_k
                        )
                    )

                    a_boundary[
                        2, k, i
                    ] = np.sum(
                        self.interior_mask
                        * (
                            v_i
                            * u_boundary_x
                            * u_k
                        )
                    )

                    a_boundary[
                        3, k, i
                    ] = np.sum(
                        self.interior_mask
                        * (
                            v_i
                            * u_boundary_y
                            * u_k
                        )
                    )

        return (
            a_equations,
            a_boundary,
        )

    def _build_linear_terms(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reproduce original LidDriven.py getB().
        """
        b_equations = np.zeros(
            (7, self.m, self.m),
            dtype=np.float64,
        )

        b_boundary = np.zeros(
            (7, self.m),
            dtype=np.float64,
        )

        (
            u_boundary_x,
            u_boundary_y,
        ) = self.first_derivatives(
            self.u_boundary
        )

        (
            u_boundary_xx,
            u_boundary_yy,
            u_boundary_xy,
        ) = self.second_derivatives(
            self.u_boundary
        )

        mode_fields = [
            self.vector_to_fields(
                self.basis[:, index]
            )
            for index in range(self.m)
        ]

        for j in range(self.m):
            p_j, u_j, v_j = mode_fields[j]

            u_j_x, u_j_y = (
                self.first_derivatives(u_j)
            )

            v_j_x, v_j_y = (
                self.first_derivatives(v_j)
            )

            p_j_x = (
                self.pressure_derivative_x(
                    p_j
                )
            )

            p_j_y = (
                self.pressure_derivative_y(
                    p_j
                )
            )

            (
                u_j_xx,
                u_j_yy,
                u_j_xy,
            ) = self.second_derivatives(u_j)

            (
                v_j_xx,
                v_j_yy,
                v_j_xy,
            ) = self.second_derivatives(v_j)

            for i in range(self.m):
                p_i, u_i, v_i = mode_fields[i]

                b_equations[
                    0, i, j
                ] = np.sum(
                    self.interior_mask
                    * (
                        u_j_x * p_i
                        + p_j_x * u_i
                    )
                )

                b_equations[
                    1, i, j
                ] = np.sum(
                    self.interior_mask
                    * (
                        u_j_y * p_i
                        + p_j_y * u_i
                    )
                )

                b_equations[
                    2, i, j
                ] = np.sum(
                    self.interior_mask
                    * (
                        v_j_x * p_i
                        + p_j_x * v_i
                    )
                )

                b_equations[
                    3, i, j
                ] = np.sum(
                    self.interior_mask
                    * (
                        v_j_y * p_i
                        + p_j_y * v_i
                    )
                )

                b_equations[
                    4, i, j
                ] = -np.sum(
                    self.interior_mask
                    * (
                        u_j_xx * u_i
                        + v_j_xx * v_i
                    )
                )

                b_equations[
                    5, i, j
                ] = -np.sum(
                    self.interior_mask
                    * (
                        u_j_yy * u_i
                        + v_j_yy * v_i
                    )
                )

                b_equations[
                    6, i, j
                ] = -np.sum(
                    self.interior_mask
                    * (
                        u_j_xy * u_i
                        + v_j_xy * v_i
                    )
                )

                b_boundary[
                    0, i
                ] = np.sum(
                    self.interior_mask
                    * u_boundary_x
                    * p_i
                )

                b_boundary[
                    1, i
                ] = np.sum(
                    self.interior_mask
                    * u_boundary_y
                    * p_i
                )

                b_boundary[
                    2, i
                ] = 0.0

                b_boundary[
                    3, i
                ] = 0.0

                b_boundary[
                    4, i
                ] = -np.sum(
                    self.interior_mask
                    * u_boundary_xx
                    * u_i
                )

                b_boundary[
                    5, i
                ] = -np.sum(
                    self.interior_mask
                    * u_boundary_yy
                    * u_i
                )

                b_boundary[
                    6, i
                ] = -np.sum(
                    self.interior_mask
                    * u_boundary_xy
                    * u_i
                )

        return (
            b_equations,
            b_boundary,
        )

    @staticmethod
    def parameter_coefficients_numpy(
        parameters: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        parameters = np.asarray(
            parameters,
            dtype=np.float64,
        )

        reynolds = parameters[:, 0:1]

        theta = (
            parameters[:, 1:2]
            * np.pi
            / 180.0
        )

        viscosity = 1.0 / reynolds

        jacobian_11 = (
            2.0 * np.ones_like(theta)
        )

        jacobian_12 = np.zeros_like(theta)

        jacobian_21 = (
            -2.0
            * np.cos(theta)
            / np.sin(theta)
        )

        jacobian_22 = (
            2.0 / np.sin(theta)
        )

        a_coefficients = np.concatenate(
            [
                jacobian_11,
                jacobian_12,
                jacobian_21,
                jacobian_22,
            ],
            axis=1,
        )

        b_coefficients = np.concatenate(
            [
                jacobian_11,
                jacobian_12,
                jacobian_21,
                jacobian_22,
                viscosity
                * (
                    jacobian_11 ** 2
                    + jacobian_21 ** 2
                ),
                viscosity
                * (
                    jacobian_12 ** 2
                    + jacobian_22 ** 2
                ),
                2.0
                * viscosity
                * (
                    jacobian_11
                    * jacobian_12
                    + jacobian_21
                    * jacobian_22
                ),
            ],
            axis=1,
        )

        return (
            a_coefficients,
            b_coefficients,
        )

    @staticmethod
    def parameter_coefficients_torch(
        parameters: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reynolds = parameters[:, 0:1]

        theta = (
            parameters[:, 1:2]
            * torch.pi
            / 180.0
        )

        viscosity = 1.0 / reynolds

        jacobian_11 = (
            2.0 * torch.ones_like(theta)
        )

        jacobian_12 = torch.zeros_like(theta)

        jacobian_21 = (
            -2.0
            * torch.cos(theta)
            / torch.sin(theta)
        )

        jacobian_22 = (
            2.0 / torch.sin(theta)
        )

        a_coefficients = torch.cat(
            [
                jacobian_11,
                jacobian_12,
                jacobian_21,
                jacobian_22,
            ],
            dim=1,
        )

        b_coefficients = torch.cat(
            [
                jacobian_11,
                jacobian_12,
                jacobian_21,
                jacobian_22,
                viscosity
                * (
                    jacobian_11 ** 2
                    + jacobian_21 ** 2
                ),
                viscosity
                * (
                    jacobian_12 ** 2
                    + jacobian_22 ** 2
                ),
                2.0
                * viscosity
                * (
                    jacobian_11
                    * jacobian_12
                    + jacobian_21
                    * jacobian_22
                ),
            ],
            dim=1,
        )

        return (
            a_coefficients,
            b_coefficients,
        )

    def assemble_numpy(
        self,
        parameters: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        a_coefficients, b_coefficients = (
            self.parameter_coefficients_numpy(
                parameters
            )
        )

        quadratic = np.einsum(
            "bq,qkij->bkij",
            a_coefficients,
            self.a_equations,
        )

        linear = np.einsum(
            "bq,qki->bki",
            a_coefficients,
            self.a_boundary,
        )

        linear += np.einsum(
            "bq,qki->bki",
            b_coefficients,
            self.b_equations,
        )

        source = -np.einsum(
            "bq,qk->bk",
            b_coefficients,
            self.b_boundary,
        )

        return (
            quadratic,
            linear,
            source,
        )

    def residual_numpy(
        self,
        outputs: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        coefficients = np.asarray(
            outputs,
            dtype=np.float64,
        )[:, :self.m]

        quadratic, linear, source = (
            self.assemble_numpy(parameters)
        )

        nonlinear_term = np.einsum(
            "bi,bkij,bj->bk",
            coefficients,
            quadratic,
            coefficients,
        )

        linear_term = np.einsum(
            "bki,bi->bk",
            linear,
            coefficients,
        )

        return (
            nonlinear_term
            + linear_term
            - source
        )

    def make_torch_residual(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ):
        a_equations = torch.as_tensor(
            self.a_equations,
            dtype=dtype,
            device=device,
        )

        a_boundary = torch.as_tensor(
            self.a_boundary,
            dtype=dtype,
            device=device,
        )

        b_equations = torch.as_tensor(
            self.b_equations,
            dtype=dtype,
            device=device,
        )

        b_boundary = torch.as_tensor(
            self.b_boundary,
            dtype=dtype,
            device=device,
        )

        m = self.m

        def residual(
            outputs: torch.Tensor,
            parameters: torch.Tensor,
        ) -> torch.Tensor:
            coefficients = outputs[:, :m]

            (
                a_coefficients,
                b_coefficients,
            ) = self.parameter_coefficients_torch(
                parameters
            )

            quadratic = torch.einsum(
                "bq,qkij->bkij",
                a_coefficients,
                a_equations,
            )

            linear = torch.einsum(
                "bq,qki->bki",
                a_coefficients,
                a_boundary,
            )

            linear = linear + torch.einsum(
                "bq,qki->bki",
                b_coefficients,
                b_equations,
            )

            source = -torch.einsum(
                "bq,qk->bk",
                b_coefficients,
                b_boundary,
            )

            nonlinear_term = torch.einsum(
                "bi,bkij,bj->bk",
                coefficients,
                quadratic,
                coefficients,
            )

            linear_term = torch.einsum(
                "bki,bi->bk",
                linear,
                coefficients,
            )

            return (
                nonlinear_term
                + linear_term
                - source
            )

        return residual

    def physical_grid(
        self,
        parameter: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        computational_x, computational_y = (
            self.chebyshev.grid()
        )

        theta = (
            float(parameter[1])
            * np.pi
            / 180.0
        )

        physical_x = (
            0.5 * computational_x
            + 0.5
            * computational_y
            * np.cos(theta)
        )

        physical_y = (
            0.5
            * computational_y
            * np.sin(theta)
        )

        return (
            physical_x,
            physical_y,
        )

    def coefficients_to_fields(
        self,
        coefficients: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        interior_vector = (
            coefficients @ self.basis.T
        )

        p_field, u_field, v_field = (
            self.vector_to_fields(
                interior_vector
            )
        )

        u_field = (
            u_field
            + self.u_boundary
        )

        return (
            p_field,
            u_field,
            v_field,
        )

# ============================================================
# 5. Errors and POD-G
# ============================================================

def mean_relative_l2_error(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    numerator = np.linalg.norm(
        truth - prediction,
        axis=1,
    )

    denominator = np.linalg.norm(
        truth,
        axis=1,
    )

    denominator = np.maximum(
        denominator,
        1.0e-14,
    )

    return float(
        np.mean(
            numerator / denominator
        )
    )

def projection_error(
    pod: PODData,
    m: int,
    validation_snapshots: np.ndarray,
) -> float:
    basis = pod.basis(m)

    coefficients = (
        validation_snapshots
        - pod.reference[None, :]
    ) @ basis

    prediction = (
        coefficients @ basis.T
        + pod.reference[None, :]
    )

    return mean_relative_l2_error(
        validation_snapshots,
        prediction,
    )

def network_error(
    trainer: SteadyPRNNTrainer,
    result: TrainResult,
    parameters: np.ndarray,
    truth: np.ndarray,
) -> float:
    prediction = trainer.predict_fields(
        result,
        parameters,
    )

    return mean_relative_l2_error(
        truth,
        prediction,
    )

def solve_pod_g(
    system: CavityReducedSystem,
    parameters: np.ndarray,
    initial_coefficients: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(
        initial_coefficients,
        dtype=np.float64,
    )

    for index in range(len(parameters)):
        parameter = parameters[
            index:index + 1
        ]

        initial = initial_coefficients[
            index
        ].copy()

        def function(
            coefficients: np.ndarray,
        ) -> np.ndarray:
            return system.residual_numpy(
                coefficients[None, :],
                parameter,
            )[0]

        solution = root(
            function,
            initial,
            method="hybr",
            options={
                "xtol": POD_G_TOLERANCE,
                "maxfev": (
                    POD_G_MAX_ITERATIONS
                    * (system.m + 1)
                ),
            },
        )

        if (
            solution.success
            and np.all(
                np.isfinite(solution.x)
            )
        ):
            output[index] = solution.x
        else:
            residual_norm = np.linalg.norm(
                function(solution.x)
            )

            print(
                f"[POD-G warning] "
                f"index={index}, "
                f"mu={parameter[0]}, "
                f"success={solution.success}, "
                f"residual={residual_norm:.4e}",
                flush=True,
            )

            output[index] = solution.x

    return output

def pod_g_error(
    system: CavityReducedSystem,
    pod: PODData,
    validation_parameters: np.ndarray,
    validation_snapshots: np.ndarray,
    cache_path: Path,
) -> float:
    basis = pod.basis(system.m)

    projection_initial = (
        validation_snapshots
        - pod.reference[None, :]
    ) @ basis

    if cache_path.exists():
        cached = np.load(
            cache_path,
            allow_pickle=False,
        )

        coefficients = np.asarray(
            cached["coefficients"],
            dtype=np.float64,
        )
    else:
        coefficients = solve_pod_g(
            system,
            validation_parameters,
            projection_initial,
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            cache_path,
            coefficients=coefficients,
            parameters=validation_parameters,
        )

    prediction = (
        coefficients @ basis.T
        + pod.reference[None, :]
    )

    return mean_relative_l2_error(
        validation_snapshots,
        prediction,
    )

# ============================================================
# 6. Training/checkpoint interface
# ============================================================

def train_or_load(
    trainer: SteadyPRNNTrainer,
    method: str,
    projection_parameters: np.ndarray,
    projection_targets: np.ndarray,
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
        omega=(
            omega
            if method == "PRNN"
            else None
        ),
    )

    if path.exists() and not force:
        print(
            f"[checkpoint] loading {path.name}",
            flush=True,
        )

        return load_checkpoint(
            path,
            device=device,
            dtype=TORCH_DTYPE,
        )

    config = TrainConfig(
        total_epochs=TOTAL_EPOCHS,
        lbfgs_epochs=LBFGS_EPOCHS,
        learning_rate=LEARNING_RATE,
        learning_rate_gamma=(
            LEARNING_RATE_GAMMA
        ),
        learning_rate_step=(
            LEARNING_RATE_STEP
        ),
        projection_batches=(
            PROJECTION_BATCHES
        ),
        residual_batches=(
            RESIDUAL_BATCHES
        ),
        pdnn_l2_lambda=(
            PDNN_L2_LAMBDA
        ),
        physics_weight=omega,
        hidden_width=HIDDEN_WIDTH,
        hidden_depth=HIDDEN_DEPTH,
        activation=ACTIVATION,
        seed=SEED,
        lbfgs_lr=1.0,
        lbfgs_history_size=50,
        lbfgs_tolerance_grad=1.0e-10,
        lbfgs_tolerance_change=1.0e-12,
        log_points_per_stage=10,
    )

    result = trainer.train(
        method=method,
        projection_parameters=(
            projection_parameters
        ),
        projection_targets=(
            projection_targets
        ),
        residual_parameters=(
            residual_parameters
        ),
        config=config,
    )

    save_checkpoint(
        path,
        result,
        metadata={
            "problem": (
                "Lid-driven cavity flow"
            ),
            "n_snapshots": n_snapshots,
            "m": m,
            "omega": omega,
            "n_residual": N_RESIDUAL,
        },
    )

    return result

# ============================================================
# 7. Plotting
# ============================================================

def plot_singular_values(
    singular_values: np.ndarray,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.0, 6.0)
    )

    indices = np.arange(
        1,
        len(singular_values) + 1,
    )

    axis.semilogy(
        indices,
        singular_values,
        color="#222222",
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )

    axis.set_xlabel(r"$m$")
    axis.set_ylabel("Singular value")

    axis.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.55,
        alpha=0.28,
    )

    figure.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.12,
        top=0.98,
    )

    save_figure(
        figure,
        (
            "Figure_1_Singular_Values_"
            f"Ns{SINGULAR_VALUE_SNAPSHOT_COUNT}.png"
        ),
    )

def plot_error_comparison(
    error_results: Dict[
        int,
        Dict[str, List[float]],
    ],
) -> None:
    """
    All snapshot counts are shown on one axis.

    Method:
        color + line style.

    Snapshot count:
        marker shape.
    """
    figure, axis = plt.subplots(
        figsize=(8.0, 6.0)
    )

    for snapshot_index, n_snapshots in enumerate(
        SNAPSHOT_COUNTS
    ):
        marker = SNAPSHOT_MARKERS[
            snapshot_index
            % len(SNAPSHOT_MARKERS)
        ]

        for method in METHODS:
            axis.semilogy(
                M_VALUES,
                error_results[
                    n_snapshots
                ][method],
                color=METHOD_COLORS[method],
                linestyle=(
                    METHOD_LINESTYLES[
                        method
                    ]
                ),
                marker=marker,
                markersize=6.0,
                markerfacecolor="none",
                markeredgewidth=1.1,
                label=(
                    f"{method}, "
                    rf"$N_s={n_snapshots}$"
                ),
            )

    axis.set_xlabel(r"$m$")
    axis.set_ylabel(
        r"Error $\varepsilon$"
    )

    axis.set_xticks(M_VALUES)

    axis.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.23,
    )

    axis.legend(
        loc="best",
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.0,
    )

    figure.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.12,
        top=0.98,
    )

    save_figure(
        figure,
        (
            "Figure_2_Error_Comparison_"
            f"{configuration_suffix()}.png"
        ),
    )

def plot_optimal_omega(
    omega_results: Dict[int, List[float]],
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.0, 6.0)
    )

    colors = [
        "#6BAED6",
        "#3182BD",
        "#08519C",
        "#08306B",
    ]

    for index, n_snapshots in enumerate(
        SNAPSHOT_COUNTS
    ):
        axis.semilogy(
            M_VALUES,
            omega_results[n_snapshots],
            color=colors[
                index % len(colors)
            ],
            linestyle="-",
            marker=SNAPSHOT_MARKERS[
                index
                % len(SNAPSHOT_MARKERS)
            ],
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=rf"$N_s={n_snapshots}$",
        )

    axis.set_xlabel(r"$m$")
    axis.set_ylabel(
        r"Quasi-optimal weight $\omega$"
    )

    axis.set_xticks(M_VALUES)
    axis.set_yticks(
        sorted(
            set(OMEGA_VALUES)
        )
    )

    axis.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.55,
        alpha=0.27,
    )

    axis.legend(
        loc="best",
        frameon=False,
    )

    figure.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.12,
        top=0.98,
    )

    save_figure(
        figure,
        (
            "Figure_3_Optimal_Omega_"
            f"{configuration_suffix()}.png"
        ),
    )

def full_sample_fields(
    full_sample: np.ndarray,
    field_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_field = np.reshape(
        full_sample[
            0::N_LOADED_VARIABLES
        ],
        field_shape,
    )

    u_field = np.reshape(
        full_sample[
            1::N_LOADED_VARIABLES
        ],
        field_shape,
    )

    v_field = np.reshape(
        full_sample[
            2::N_LOADED_VARIABLES
        ],
        field_shape,
    )

    return (
        p_field,
        u_field,
        v_field,
    )

def plot_truth_prnn_fields(
    system: CavityReducedSystem,
    parameters: np.ndarray,
    truth_full_samples: np.ndarray,
    predicted_coefficients: np.ndarray,
    n_snapshots: int,
    m: int,
) -> None:
    n_cases = len(parameters)

    figure, axes = plt.subplots(
        2,
        n_cases,
        figsize=(8.0, 6.0),
        squeeze=False,
    )

    truth_fields = []
    prediction_fields = []

    for index in range(n_cases):
        truth_fields.append(
            full_sample_fields(
                truth_full_samples[:, index],
                system.field_shape,
            )
        )

        prediction_fields.append(
            system.coefficients_to_fields(
                predicted_coefficients[index]
            )
        )

    pressure_minimum = min(
        np.min(fields[0])
        for fields in (
            truth_fields
            + prediction_fields
        )
    )

    pressure_maximum = max(
        np.max(fields[0])
        for fields in (
            truth_fields
            + prediction_fields
        )
    )

    pressure_levels = np.linspace(
        pressure_minimum,
        pressure_maximum,
        31,
    )

    image = None

    for column in range(n_cases):
        parameter = parameters[column]

        physical_x, physical_y = (
            system.physical_grid(
                parameter
            )
        )

        rows = [
            truth_fields[column],
            prediction_fields[column],
        ]

        for row, fields in enumerate(rows):
            pressure, velocity_u, velocity_v = fields

            axis = axes[row, column]

            image = axis.contourf(
                physical_x,
                physical_y,
                pressure,
                levels=pressure_levels,
                cmap="RdYlBu_r",
                extend="both",
            )

            stride = max(
                1,
                system.field_shape[0] // 12,
            )

            axis.quiver(
                physical_x[
                    ::stride,
                    ::stride,
                ],
                physical_y[
                    ::stride,
                    ::stride,
                ],
                velocity_u[
                    ::stride,
                    ::stride,
                ],
                velocity_v[
                    ::stride,
                    ::stride,
                ],
                color="black",
                angles="xy",
                scale_units="xy",
                scale=None,
                width=0.004,
                alpha=0.80,
            )

            axis.set_aspect(
                "equal",
                adjustable="box",
            )

            axis.set_xticks([])
            axis.set_yticks([])

        reynolds = parameter[0]
        theta_degrees = parameter[1]

        axes[0, column].set_title(
            rf"$\boldsymbol{{\mu}}="
            rf"({reynolds:.0f},"
            rf"{theta_degrees:.0f}^\circ)$"
        )

    axes[0, 0].set_ylabel("Truth")
    axes[1, 0].set_ylabel("PRNN")

    color_axis = figure.add_axes(
        [0.925, 0.10, 0.018, 0.82]
    )

    figure.colorbar(
        image,
        cax=color_axis,
        label=r"Pressure $p$",
    )

    figure.subplots_adjust(
        left=0.055,
        right=0.90,
        bottom=0.045,
        top=0.94,
        wspace=0.06,
        hspace=0.08,
    )

    save_figure(
        figure,
        (
            "Figure_4_Truth_PRNN_"
            f"Ns{n_snapshots}_m{m}.png"
        ),
    )

# ============================================================
# 8. CSV and metadata
# ============================================================

def write_results_csv(
    error_results: Dict[
        int,
        Dict[str, List[float]],
    ],
    omega_results: Dict[int, List[float]],
) -> None:
    path = (
        RESULT_DIR
        / "numerical_results.csv"
    )

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
            for index, m in enumerate(
                M_VALUES
            ):
                writer.writerow(
                    [
                        n_snapshots,
                        m,
                        error_results[
                            n_snapshots
                        ]["Projection"][index],
                        error_results[
                            n_snapshots
                        ]["POD-G"][index],
                        error_results[
                            n_snapshots
                        ]["PDNN"][index],
                        error_results[
                            n_snapshots
                        ]["PINN"][index],
                        error_results[
                            n_snapshots
                        ]["PRNN"][index],
                        omega_results[
                            n_snapshots
                        ][index],
                    ]
                )

# ============================================================
# 9. Main experiment
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Steady lid-driven cavity "
            "PDNN/PINN/PRNN experiment."
        )
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
        help=(
            "Training device. 'auto' selects CUDA "
            "when available, otherwise CPU."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Retrain all networks and overwrite "
            "existing checkpoints."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 9.1 Initialization
    # --------------------------------------------------------

    configure_plotting()
    set_seed(SEED)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but CUDA is unavailable."
        )

    if args.device == "cpu":
        device_object = torch.device("cpu")
    elif args.device == "cuda":
        device_object = torch.device("cuda")
    else:
        device_object = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 9.2 Locate and load original open-source data
    # --------------------------------------------------------

    pod_file = find_file_recursively(
        ROOT,
        POD_FILE_NAME,
    )

    validation_file = find_file_recursively(
        ROOT,
        VALIDATION_FILE_NAME,
    )

    cavity_data = load_cavity_data(
        pod_file,
        validation_file,
    )

    print("=" * 80)
    print("Lid-driven cavity PRNN experiment")
    print(f"Root                 : {ROOT}")
    print(f"POD file             : {pod_file}")
    print(f"Validation/test file : {validation_file}")
    print(f"Results directory    : {RESULT_DIR}")
    print(f"Field shape          : {cavity_data.field_shape}")
    print(f"Snapshot counts      : {SNAPSHOT_COUNTS}")
    print(f"POD dimensions       : {M_VALUES}")
    print(f"Residual points      : {N_RESIDUAL}")
    print(f"Omega candidates     : {OMEGA_VALUES}")
    print(
        "Parameter domain      : "
        f"[{PARAMETER_LOWER[0]}, {PARAMETER_UPPER[0]}]"
        " x "
        f"[{PARAMETER_LOWER[1]}, {PARAMETER_UPPER[1]}] degrees"
    )
    print(
        "Network width/depth  : "
        f"{HIDDEN_WIDTH}/{HIDDEN_DEPTH}"
    )
    print(f"Activation           : {ACTIVATION}")
    print(f"Total epochs         : {TOTAL_EPOCHS}")
    print(f"Adam epochs          : {TOTAL_EPOCHS - LBFGS_EPOCHS}")
    print(f"L-BFGS epochs        : {LBFGS_EPOCHS}")
    print(f"Learning rate        : {LEARNING_RATE}")
    print(
        "Learning-rate decay : "
        f"{LEARNING_RATE_GAMMA}"
        f" every {LEARNING_RATE_STEP} epochs"
    )
    print(f"PDNN L2 lambda       : {PDNN_L2_LAMBDA}")
    print(f"Device               : {device_object}")
    print(f"Torch dtype          : {TORCH_DTYPE}")
    print(f"Force retraining     : {args.force}")
    print("=" * 80)

    if cavity_data.samples.shape[1] < max(
        SNAPSHOT_COUNTS
    ):
        raise RuntimeError(
            "The POD MAT file contains only "
            f"{cavity_data.samples.shape[1]} snapshots, "
            f"but Ns={max(SNAPSHOT_COUNTS)} is required."
        )

    if cavity_data.parameters.shape[0] < max(
        SNAPSHOT_COUNTS
    ):
        raise RuntimeError(
            "The POD parameter array contains only "
            f"{cavity_data.parameters.shape[0]} rows, "
            f"but Ns={max(SNAPSHOT_COUNTS)} is required."
        )

    if len(cavity_data.validation_parameters) == 0:
        raise RuntimeError(
            "The validation/test parameter set is empty."
        )

    # --------------------------------------------------------
    # 9.3 Extract the original interior p-u-v state
    # --------------------------------------------------------

    all_training_snapshots = (
        extract_interior_snapshots(
            cavity_data.samples,
            cavity_data.field_shape,
        ).T
    )

    validation_snapshots = (
        extract_interior_snapshots(
            cavity_data.validation_samples,
            cavity_data.field_shape,
        ).T
    )

    validation_parameters_all = np.asarray(
        cavity_data.validation_parameters,
        dtype=np.float64,
    )

    if (
        len(validation_snapshots)
        != len(validation_parameters_all)
    ):
        raise RuntimeError(
            "Validation snapshots and validation parameters "
            "have inconsistent sizes."
        )

    print(
        "\n[data] Full training snapshot array: "
        f"{all_training_snapshots.shape}"
    )

    print(
        "[data] Validation/test snapshot array: "
        f"{validation_snapshots.shape}"
    )

    print(
        "[data] Validation/test parameters: "
        f"{validation_parameters_all.shape}"
    )

    # --------------------------------------------------------
    # 9.4 Generate Latin-hypercube residual parameters
    # --------------------------------------------------------

    residual_sampler = qmc.LatinHypercube(
        d=len(PARAMETER_LOWER),
        seed=SEED + 1,
    )

    residual_parameters = qmc.scale(
        residual_sampler.random(
            n=N_RESIDUAL,
        ),
        PARAMETER_LOWER,
        PARAMETER_UPPER,
    )

    np.savez_compressed(
        DATA_DIR / "residual_parameters.npz",
        parameters=residual_parameters,
        lower=PARAMETER_LOWER,
        upper=PARAMETER_UPPER,
        seed=np.asarray([SEED + 1]),
    )

    print(
        "[sampling] Residual parameter array: "
        f"{residual_parameters.shape}"
    )

    # --------------------------------------------------------
    # 9.5 Select 50 validation points for omega selection
    # --------------------------------------------------------
    #
    # The supplied repository contains one high-fidelity
    # validation/test file. We therefore use a deterministic
    # 50-point subset only for model/omega selection, while all
    # available points are used for reporting final errors.
    # --------------------------------------------------------

    validation_rng = np.random.default_rng(
        SEED + 2
    )

    validation_count = min(
        N_VALIDATION_SELECTION,
        len(validation_parameters_all),
    )

    validation_selection_indices = np.sort(
        validation_rng.choice(
            len(validation_parameters_all),
            size=validation_count,
            replace=False,
        )
    )

    selection_parameters = (
        validation_parameters_all[
            validation_selection_indices
        ]
    )

    selection_snapshots = (
        validation_snapshots[
            validation_selection_indices
        ]
    )

    np.savez_compressed(
        DATA_DIR / "omega_selection_set.npz",
        indices=validation_selection_indices,
        parameters=selection_parameters,
        snapshots=selection_snapshots,
        seed=np.asarray([SEED + 2]),
    )

    print(
        "[sampling] Omega-selection set: "
        f"{validation_count} points"
    )

    print(
        "[sampling] Final reported test set: "
        f"{len(validation_parameters_all)} points"
    )

    # --------------------------------------------------------
    # 9.6 Construct POD data for each Ns
    # --------------------------------------------------------

    pod_by_snapshot_count: Dict[
        int,
        PODData,
    ] = {}

    for n_snapshots in SNAPSHOT_COUNTS:
        snapshot_parameters = np.asarray(
            cavity_data.parameters[
                :n_snapshots
            ],
            dtype=np.float64,
        )

        snapshot_states = np.asarray(
            all_training_snapshots[
                :n_snapshots
            ],
            dtype=np.float64,
        )

        pod = build_pod(
            parameters=snapshot_parameters,
            snapshots=snapshot_states,
            reference_mode="zero",
        )

        pod_by_snapshot_count[
            n_snapshots
        ] = pod

        np.savez_compressed(
            DATA_DIR
            / f"POD_Ns{n_snapshots}.npz",
            parameters=pod.parameters,
            snapshots=pod.snapshots,
            reference=pod.reference,
            basis_full=pod.basis_full,
            singular_values=pod.singular_values,
            projection_coefficients_full=(
                pod.projection_coefficients_full
            ),
        )

        print(
            f"[POD] Ns={n_snapshots}: "
            f"snapshot matrix={pod.snapshot_matrix.shape}, "
            f"available modes={pod.basis_full.shape[1]}"
        )

        print(
            f"[POD] Ns={n_snapshots}: "
            f"first singular values="
            f"{pod.singular_values[:min(10, len(pod.singular_values))]}"
        )

        if max(M_VALUES) > pod.basis_full.shape[1]:
            raise RuntimeError(
                f"Ns={n_snapshots} provides only "
                f"{pod.basis_full.shape[1]} POD modes, "
                f"but m={max(M_VALUES)} was requested."
            )

    if (
        SINGULAR_VALUE_SNAPSHOT_COUNT
        not in pod_by_snapshot_count
    ):
        raise RuntimeError(
            "SINGULAR_VALUE_SNAPSHOT_COUNT must be one "
            "of SNAPSHOT_COUNTS."
        )

    plot_singular_values(
        pod_by_snapshot_count[
            SINGULAR_VALUE_SNAPSHOT_COUNT
        ].singular_values
    )

    print(
        "[figure] Saved singular-value figure for "
        f"Ns={SINGULAR_VALUE_SNAPSHOT_COUNT}."
    )

    # --------------------------------------------------------
    # 9.7 Result containers
    # --------------------------------------------------------

    error_results: Dict[
        int,
        Dict[str, List[float]],
    ] = {}

    omega_results: Dict[
        int,
        List[float],
    ] = {}

    validation_error_results: Dict[
        int,
        Dict[int, Dict[float, float]],
    ] = {}

    selected_prnn: Dict[
        Tuple[int, int],
        TrainResult,
    ] = {}

    selected_trainers: Dict[
        Tuple[int, int],
        SteadyPRNNTrainer,
    ] = {}

    selected_systems: Dict[
        Tuple[int, int],
        CavityReducedSystem,
    ] = {}

    selected_pods: Dict[
        Tuple[int, int],
        PODData,
    ] = {}

    training_histories: Dict[
        str,
        List[Dict],
    ] = {}

    # --------------------------------------------------------
    # 9.8 Main Ns-m experiment loop
    # --------------------------------------------------------

    for n_snapshots in SNAPSHOT_COUNTS:
        pod = pod_by_snapshot_count[
            n_snapshots
        ]

        error_results[n_snapshots] = {
            method: []
            for method in METHODS
        }

        omega_results[n_snapshots] = []

        validation_error_results[
            n_snapshots
        ] = {}

        print("\n" + "=" * 80)
        print(f"Snapshot count Ns = {n_snapshots}")
        print("=" * 80)

        for m in M_VALUES:
            print("\n" + "-" * 80)
            print(f"Ns={n_snapshots}, m={m}")
            print("-" * 80)

            validation_error_results[
                n_snapshots
            ][m] = {}

            # ------------------------------------------------
            # Construct the cavity-specific reduced residual
            # ------------------------------------------------

            system = CavityReducedSystem(
                cavity_data=cavity_data,
                pod=pod,
                m=m,
            )

            residual_torch = (
                system.make_torch_residual(
                    device=device_object,
                    dtype=TORCH_DTYPE,
                )
            )

            problem = SteadyReducedProblem(
                basis=pod.basis(m),
                reference=pod.reference,
                input_lower=PARAMETER_LOWER,
                input_upper=PARAMETER_UPPER,
                coefficient_dimension=m,
                residual_torch=residual_torch,
                residual_numpy=(
                    system.residual_numpy
                ),
                name=(
                    "Steady lid-driven cavity flow"
                ),
            )

            trainer = SteadyPRNNTrainer(
                problem=problem,
                device=args.device,
                dtype=TORCH_DTYPE,
            )

            projection_targets = (
                pod.projection_coefficients(m)
            )

            # ------------------------------------------------
            # Projection error
            # ------------------------------------------------

            projection_value = projection_error(
                pod=pod,
                m=m,
                validation_snapshots=(
                    validation_snapshots
                ),
            )

            # ------------------------------------------------
            # POD-G error
            # ------------------------------------------------
            #
            # Following the original open-source experiment,
            # the high-fidelity projection at each test point
            # is used as the nonlinear-solver initial value.
            # ------------------------------------------------

            pod_g_value = pod_g_error(
                system=system,
                pod=pod,
                validation_parameters=(
                    validation_parameters_all
                ),
                validation_snapshots=(
                    validation_snapshots
                ),
                cache_path=(
                    DATA_DIR
                    / (
                        f"PODG_Ns{n_snapshots}_"
                        f"m{m}.npz"
                    )
                ),
            )

            # ------------------------------------------------
            # PDNN
            # ------------------------------------------------

            pdnn_result = train_or_load(
                trainer=trainer,
                method="PDNN",
                projection_parameters=(
                    pod.parameters
                ),
                projection_targets=(
                    projection_targets
                ),
                residual_parameters=(
                    residual_parameters
                ),
                n_snapshots=n_snapshots,
                m=m,
                omega=1.0,
                device=args.device,
                force=args.force,
            )

            pdnn_value = network_error(
                trainer=trainer,
                result=pdnn_result,
                parameters=(
                    validation_parameters_all
                ),
                truth=validation_snapshots,
            )

            training_histories[
                f"PDNN_Ns{n_snapshots}_m{m}"
            ] = pdnn_result.history

            # ------------------------------------------------
            # PINN
            # ------------------------------------------------

            pinn_result = train_or_load(
                trainer=trainer,
                method="PINN",
                projection_parameters=(
                    pod.parameters
                ),
                projection_targets=(
                    projection_targets
                ),
                residual_parameters=(
                    residual_parameters
                ),
                n_snapshots=n_snapshots,
                m=m,
                omega=1.0,
                device=args.device,
                force=args.force,
            )

            pinn_value = network_error(
                trainer=trainer,
                result=pinn_result,
                parameters=(
                    validation_parameters_all
                ),
                truth=validation_snapshots,
            )

            training_histories[
                f"PINN_Ns{n_snapshots}_m{m}"
            ] = pinn_result.history

            # ------------------------------------------------
            # PRNN grid search over omega
            # ------------------------------------------------

            best_prnn_result: Optional[
                TrainResult
            ] = None

            best_validation_error = math.inf
            best_omega = float(
                OMEGA_VALUES[0]
            )

            for omega in OMEGA_VALUES:
                candidate = train_or_load(
                    trainer=trainer,
                    method="PRNN",
                    projection_parameters=(
                        pod.parameters
                    ),
                    projection_targets=(
                        projection_targets
                    ),
                    residual_parameters=(
                        residual_parameters
                    ),
                    n_snapshots=n_snapshots,
                    m=m,
                    omega=float(omega),
                    device=args.device,
                    force=args.force,
                )

                candidate_validation_error = (
                    network_error(
                        trainer=trainer,
                        result=candidate,
                        parameters=(
                            selection_parameters
                        ),
                        truth=selection_snapshots,
                    )
                )

                validation_error_results[
                    n_snapshots
                ][m][float(omega)] = (
                    candidate_validation_error
                )

                training_histories[
                    (
                        f"PRNN_Ns{n_snapshots}_"
                        f"m{m}_omega{omega:.0e}"
                    )
                ] = candidate.history

                print(
                    f"[validation] "
                    f"Ns={n_snapshots}, "
                    f"m={m}, "
                    f"omega={omega:.2e}, "
                    f"error="
                    f"{candidate_validation_error:.6e}",
                    flush=True,
                )

                if (
                    candidate_validation_error
                    < best_validation_error
                ):
                    best_validation_error = (
                        candidate_validation_error
                    )

                    best_prnn_result = candidate
                    best_omega = float(omega)

            if best_prnn_result is None:
                raise RuntimeError(
                    f"No PRNN candidate was selected for "
                    f"Ns={n_snapshots}, m={m}."
                )

            best_prnn_result.best_validation_error = (
                best_validation_error
            )

            prnn_value = network_error(
                trainer=trainer,
                result=best_prnn_result,
                parameters=(
                    validation_parameters_all
                ),
                truth=validation_snapshots,
            )

            # ------------------------------------------------
            # Store final errors
            # ------------------------------------------------

            error_results[
                n_snapshots
            ]["Projection"].append(
                projection_value
            )

            error_results[
                n_snapshots
            ]["POD-G"].append(
                pod_g_value
            )

            error_results[
                n_snapshots
            ]["PDNN"].append(
                pdnn_value
            )

            error_results[
                n_snapshots
            ]["PINN"].append(
                pinn_value
            )

            error_results[
                n_snapshots
            ]["PRNN"].append(
                prnn_value
            )

            omega_results[
                n_snapshots
            ].append(
                best_omega
            )

            selected_prnn[
                (n_snapshots, m)
            ] = best_prnn_result

            selected_trainers[
                (n_snapshots, m)
            ] = trainer

            selected_systems[
                (n_snapshots, m)
            ] = system

            selected_pods[
                (n_snapshots, m)
            ] = pod

            print(
                "[error] "
                f"Projection={projection_value:.6e}, "
                f"POD-G={pod_g_value:.6e}, "
                f"PDNN={pdnn_value:.6e}, "
                f"PINN={pinn_value:.6e}, "
                f"PRNN={prnn_value:.6e}, "
                f"omega={best_omega:.2e}",
                flush=True,
            )

    # --------------------------------------------------------
    # 9.9 Save numerical tables before plotting
    # --------------------------------------------------------

    write_results_csv(
        error_results,
        omega_results,
    )

    validation_grid_search_path = (
        RESULT_DIR
        / "prnn_validation_grid_search.json"
    )

    validation_grid_search_serializable = {
        str(n_snapshots): {
            str(m): {
                f"{omega:.12g}": error
                for omega, error
                in validation_error_results[
                    n_snapshots
                ][m].items()
            }
            for m in M_VALUES
        }
        for n_snapshots in SNAPSHOT_COUNTS
    }

    validation_grid_search_path.write_text(
        json.dumps(
            validation_grid_search_serializable,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    training_history_path = (
        RESULT_DIR
        / "training_histories.json"
    )

    training_history_path.write_text(
        json.dumps(
            training_histories,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n[data] Numerical results saved before plotting."
    )

    # --------------------------------------------------------
    # 9.10 Figure 2: all Ns on one error figure
    # --------------------------------------------------------

    plot_error_comparison(
        error_results
    )

    print(
        "[figure] Saved combined error comparison."
    )

    # --------------------------------------------------------
    # 9.11 Figure 3: quasi-optimal omega
    # --------------------------------------------------------

    plot_optimal_omega(
        omega_results
    )

    print(
        "[figure] Saved quasi-optimal omega comparison."
    )

    # --------------------------------------------------------
    # 9.12 Figure 4: truth versus PRNN with vector overlay
    # --------------------------------------------------------

    available_field_keys = sorted(
        set(selected_prnn.keys())
        & set(selected_trainers.keys())
        & set(selected_systems.keys())
        & set(selected_pods.keys())
    )

    if not available_field_keys:
        raise RuntimeError(
            "No trained PRNN model is available "
            "for Figure 4."
        )

    # Use the largest available Ns.
    field_snapshot_count = max(
        n_snapshots
        for n_snapshots, _ in (
            available_field_keys
        )
    )

    # Use the largest available m under that Ns.
    field_m = max(
        m
        for n_snapshots, m in (
            available_field_keys
        )
        if (
            n_snapshots
            == field_snapshot_count
        )
    )

    field_key = (
        field_snapshot_count,
        field_m,
    )

    n_validation_cases = len(
        validation_parameters_all
    )

    valid_preferred_indices = [
        index
        for index in PREFERRED_FIELD_INDICES
        if 0 <= index < n_validation_cases
    ]

    if len(valid_preferred_indices) >= 3:
        field_indices = np.asarray(
            valid_preferred_indices[:3],
            dtype=int,
        )
    elif n_validation_cases >= 3:
        field_indices = np.linspace(
            0,
            n_validation_cases - 1,
            3,
            dtype=int,
        )
    else:
        field_indices = np.arange(
            n_validation_cases,
            dtype=int,
        )

    field_parameters = (
        validation_parameters_all[
            field_indices
        ]
    )

    field_truth_full_samples = (
        cavity_data.validation_samples[
            :,
            field_indices,
        ]
    )

    field_result = selected_prnn[
        field_key
    ]

    field_trainer = selected_trainers[
        field_key
    ]

    field_system = selected_systems[
        field_key
    ]

    field_coefficients = (
        field_trainer.predict_coefficients(
            field_result,
            field_parameters,
        )
    )

    print(
        "\n[field figure] "
        f"Ns={field_snapshot_count}, "
        f"m={field_m}, "
        f"indices={field_indices.tolist()}"
    )

    for index, parameter in zip(
        field_indices,
        field_parameters,
    ):
        print(
            "[field figure] "
            f"index={int(index)}, "
            f"Re={parameter[0]:.6g}, "
            f"theta={parameter[1]:.6g} degrees"
        )

    plot_truth_prnn_fields(
        system=field_system,
        parameters=field_parameters,
        truth_full_samples=(
            field_truth_full_samples
        ),
        predicted_coefficients=(
            field_coefficients
        ),
        n_snapshots=field_snapshot_count,
        m=field_m,
    )

    print(
        "[figure] Saved truth/PRNN pressure-vector comparison."
    )

    # --------------------------------------------------------
    # 9.13 Save complete metadata
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    metadata = {
        "problem": (
            "Steady lid-driven cavity flow"
        ),
        "root": str(ROOT),
        "results_directory": str(
            RESULT_DIR
        ),
        "pod_file": str(pod_file),
        "validation_file": str(
            validation_file
        ),
        "field_shape": list(
            cavity_data.field_shape
        ),
        "state_definition": (
            "Interior p, u, v variables in the original "
            "open-source interleaved order."
        ),
        "reference_state": "zero",
        "snapshot_counts": (
            SNAPSHOT_COUNTS
        ),
        "singular_value_snapshot_count": (
            SINGULAR_VALUE_SNAPSHOT_COUNT
        ),
        "m_values": M_VALUES,
        "omega_values": OMEGA_VALUES,
        "n_residual": N_RESIDUAL,
        "n_omega_selection": (
            validation_count
        ),
        "n_reported_test": int(
            len(validation_parameters_all)
        ),
        "parameter_definition": [
            "Reynolds number Re",
            "cavity angle theta in degrees",
        ],
        "parameter_lower": (
            PARAMETER_LOWER.tolist()
        ),
        "parameter_upper": (
            PARAMETER_UPPER.tolist()
        ),
        "network": {
            "hidden_width": HIDDEN_WIDTH,
            "hidden_depth": HIDDEN_DEPTH,
            "activation": ACTIVATION,
            "input_dimension": 2,
            "output_dimension": (
                "equal to POD dimension m"
            ),
        },
        "training": {
            "total_epochs": TOTAL_EPOCHS,
            "adam_epochs": (
                TOTAL_EPOCHS
                - LBFGS_EPOCHS
            ),
            "lbfgs_epochs": (
                LBFGS_EPOCHS
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "learning_rate_gamma": (
                LEARNING_RATE_GAMMA
            ),
            "learning_rate_step": (
                LEARNING_RATE_STEP
            ),
            "projection_batches": (
                PROJECTION_BATCHES
            ),
            "residual_batches": (
                RESIDUAL_BATCHES
            ),
            "pdnn_l2_lambda": (
                PDNN_L2_LAMBDA
            ),
            "seed": SEED,
            "torch_dtype": str(
                TORCH_DTYPE
            ),
            "device": str(
                device_object
            ),
        },
        "field_figure": {
            "n_snapshots": (
                field_snapshot_count
            ),
            "m": field_m,
            "indices": (
                field_indices.tolist()
            ),
            "parameters": (
                field_parameters.tolist()
            ),
        },
        "optimal_omega": {
            str(key): value
            for key, value
            in omega_results.items()
        },
        "errors": {
            str(key): value
            for key, value
            in error_results.items()
        },
        "validation_grid_search": (
            validation_grid_search_serializable
        ),
        "elapsed_seconds": elapsed,
        "output_files": {
            "singular_values": (
                "figures/"
                f"Figure_1_Singular_Values_"
                f"Ns{SINGULAR_VALUE_SNAPSHOT_COUNT}.png"
            ),
            "error_comparison": (
                "figures/"
                "Figure_2_Error_Comparison_"
                f"{configuration_suffix()}.png"
            ),
            "optimal_omega": (
                "figures/"
                "Figure_3_Optimal_Omega_"
                f"{configuration_suffix()}.png"
            ),
            "truth_prnn": (
                "figures/"
                "Figure_4_Truth_PRNN_"
                f"Ns{field_snapshot_count}_"
                f"m{field_m}.png"
            ),
            "numerical_results": (
                "numerical_results.csv"
            ),
            "validation_grid_search": (
                validation_grid_search_path.name
            ),
            "training_histories": (
                training_history_path.name
            ),
        },
        "notes": [
            (
                "The model is steady-state, so no time derivative "
                "of the neural-network reduced coefficients is used."
            ),
            (
                "The POD reference field is zero, following the "
                "paper's steady-state setting."
            ),
            (
                "The input parameters are normalized componentwise "
                "to [-1, 1]."
            ),
            (
                "The POD coefficients are normalized componentwise "
                "using their mean and standard deviation."
            ),
            (
                "PDNN uses the normalized projection-coefficient "
                "MSE plus lambda times the squared Frobenius norm "
                "of the Linear-layer weight matrices."
            ),
            (
                "PINN minimizes the mean squared reduced "
                "POD-G residual."
            ),
            (
                "PRNN minimizes the PDNN objective plus omega "
                "times the PINN residual loss."
            ),
            (
                "POD-G uses the high-fidelity projection at each "
                "reported test parameter as its nonlinear-solver "
                "initial value, matching the original cavity code."
            ),
            (
                "The supplied repository contains one high-fidelity "
                "validation/test MAT file. A fixed random subset is "
                "used for omega selection, while all available "
                "points are used for the final reported errors."
            ),
            (
                "Figure 2 places all snapshot counts on one "
                "semilogarithmic error axis."
            ),
            (
                "Figure 4 uses pressure contours with velocity "
                "vector overlays in the physical parallelogram."
            ),
            (
                "With M_VALUES=[2,4,6,8], Figure 4 dynamically "
                "uses Ns=100 and m=8."
            ),
        ],
    }

    metadata_path = (
        RESULT_DIR
        / "experiment_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 9.14 Final console summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Lid-driven cavity experiment completed")
    print(f"Results directory : {RESULT_DIR}")
    print(f"Figures directory : {FIGURE_DIR}")
    print(
        "CSV               : "
        f"{RESULT_DIR / 'numerical_results.csv'}"
    )
    print(
        "Omega validation  : "
        f"{validation_grid_search_path}"
    )
    print(
        "Training history  : "
        f"{training_history_path}"
    )
    print(f"Metadata          : {metadata_path}")
    print(f"Elapsed seconds   : {elapsed:.2f}")
    print("=" * 80)

    print("\nFinal mean relative L2 errors")

    for n_snapshots in SNAPSHOT_COUNTS:
        print("\n" + "-" * 80)
        print(f"Ns = {n_snapshots}")
        print("-" * 80)

        for index, m in enumerate(
            M_VALUES
        ):
            print(
                f"m={m:2d}: "
                f"Projection="
                f"{error_results[n_snapshots]['Projection'][index]:.6e}, "
                f"POD-G="
                f"{error_results[n_snapshots]['POD-G'][index]:.6e}, "
                f"PDNN="
                f"{error_results[n_snapshots]['PDNN'][index]:.6e}, "
                f"PINN="
                f"{error_results[n_snapshots]['PINN'][index]:.6e}, "
                f"PRNN="
                f"{error_results[n_snapshots]['PRNN'][index]:.6e}, "
                f"omega="
                f"{omega_results[n_snapshots][index]:.2e}"
            )

    print("\nGenerated figures")

    figure_names = [
        (
            "Figure_1_Singular_Values_"
            f"Ns{SINGULAR_VALUE_SNAPSHOT_COUNT}.png"
        ),
        (
            "Figure_2_Error_Comparison_"
            f"{configuration_suffix()}.png"
        ),
        (
            "Figure_3_Optimal_Omega_"
            f"{configuration_suffix()}.png"
        ),
        (
            "Figure_4_Truth_PRNN_"
            f"Ns{field_snapshot_count}_"
            f"m{field_m}.png"
        ),
    ]

    for figure_name in figure_names:
        print(
            f"  {FIGURE_DIR / figure_name}"
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
