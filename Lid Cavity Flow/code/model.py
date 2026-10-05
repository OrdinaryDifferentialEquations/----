from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor

TorchResidual = Callable[[Tensor, Tensor], Tensor]
NumpyResidual = Callable[[np.ndarray, np.ndarray], np.ndarray]
ActivationSpecification = Union[
    str,
    Callable[[], nn.Module],
]

# ============================================================
# 1. Reproducibility and device
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(
    preference: str = "auto",
) -> torch.device:
    preference = preference.lower()

    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is unavailable."
            )

        return torch.device("cuda")

    if preference != "auto":
        raise ValueError(
            "device must be one of: auto, cpu, cuda."
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

# ============================================================
# 2. Input and output normalization
# ============================================================

def normalize_inputs(
    values: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray:
    """
    Paper equation (27):

        x_tilde =
            [x - (x_max + x_min)/2]
            / [(x_max - x_min)/2].

    This maps each input component to [-1, 1].
    """
    values = np.asarray(values, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError(
            "values must have shape [n_samples, input_dimension]."
        )

    if lower.shape != upper.shape:
        raise ValueError(
            "lower and upper must have identical shapes."
        )

    if values.shape[1] != lower.size:
        raise ValueError(
            "Input dimension is inconsistent with bounds."
        )

    half_range = 0.5 * (upper - lower)

    if np.any(half_range <= 0.0):
        raise ValueError(
            "Every upper bound must be larger than its lower bound."
        )

    midpoint = 0.5 * (upper + lower)

    return (values - midpoint[None, :]) / half_range[None, :]

@dataclass
class StandardScaling:
    """
    Componentwise standard-score normalization:

        y_tilde_j = (y_j - mean_j) / std_j.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        minimum_std: float = 1.0e-12,
    ) -> "StandardScaling":
        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 2:
            raise ValueError(
                "Scaling values must have shape "
                "[n_samples, output_dimension]."
            )

        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std = np.maximum(std, minimum_std)

        return cls(
            mean=mean,
            std=std,
        )

    def normalize(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (values - self.mean[None, :]) / self.std[None, :]

    def denormalize(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (
            values * self.std[None, :]
            + self.mean[None, :]
        )

# ============================================================
# 3. POD construction
# ============================================================

@dataclass
class PODData:
    """
    POD data for a steady-state problem.

    snapshots:
        Shape [n_snapshots, full_dimension].

    snapshot_matrix:
        Shape [full_dimension, n_snapshots].

    basis_full:
        All available left singular vectors.

    projection_coefficients_full:
        Projection onto all available POD modes.
    """

    parameters: np.ndarray
    snapshots: np.ndarray
    snapshot_matrix: np.ndarray
    reference: np.ndarray
    centered_snapshots: np.ndarray
    basis_full: np.ndarray
    singular_values: np.ndarray
    projection_coefficients_full: np.ndarray

    def basis(
        self,
        m: int,
    ) -> np.ndarray:
        if m <= 0:
            raise ValueError("m must be positive.")

        if m > self.basis_full.shape[1]:
            raise ValueError(
                f"Requested m={m}, but only "
                f"{self.basis_full.shape[1]} POD modes exist."
            )

        return self.basis_full[:, :m]

    def projection_coefficients(
        self,
        m: int,
    ) -> np.ndarray:
        return self.projection_coefficients_full[:, :m]

    def reconstruct(
        self,
        coefficients: np.ndarray,
        m: Optional[int] = None,
    ) -> np.ndarray:
        coefficients = np.asarray(
            coefficients,
            dtype=np.float64,
        )

        if m is None:
            m = coefficients.shape[1]

        basis = self.basis(m)

        return (
            coefficients @ basis.T
            + self.reference[None, :]
        )

def build_pod(
    parameters: np.ndarray,
    snapshots: np.ndarray,
    reference: Optional[np.ndarray] = None,
    reference_mode: str = "zero",
) -> PODData:
    """
    Construct a steady-state POD basis.

    Parameters
    ----------
    parameters:
        Shape [Ns, parameter_dimension].

    snapshots:
        Shape [Ns, full_dimension].

    reference:
        Optional explicitly supplied reference state.

    reference_mode:
        Used only when reference is None.

        "zero":
            reference = 0.

        "mean":
            reference = mean of snapshots.
    """
    parameters = np.asarray(
        parameters,
        dtype=np.float64,
    )

    snapshots = np.asarray(
        snapshots,
        dtype=np.float64,
    )

    if parameters.ndim != 2:
        raise ValueError(
            "parameters must have shape [Ns, parameter_dimension]."
        )

    if snapshots.ndim != 2:
        raise ValueError(
            "snapshots must have shape [Ns, full_dimension]."
        )

    if len(parameters) != len(snapshots):
        raise ValueError(
            "parameters and snapshots must contain "
            "the same number of samples."
        )

    full_dimension = snapshots.shape[1]

    if reference is None:
        mode = reference_mode.lower()

        if mode == "zero":
            reference = np.zeros(
                full_dimension,
                dtype=np.float64,
            )
        elif mode == "mean":
            reference = np.mean(
                snapshots,
                axis=0,
            )
        else:
            raise ValueError(
                "reference_mode must be zero or mean."
            )
    else:
        reference = np.asarray(
            reference,
            dtype=np.float64,
        ).reshape(-1)

    if reference.size != full_dimension:
        raise ValueError(
            "reference dimension is inconsistent with snapshots."
        )

    centered = snapshots - reference[None, :]
    snapshot_matrix = centered.T

    basis_full, singular_values, _ = np.linalg.svd(
        snapshot_matrix,
        full_matrices=False,
    )

    projection_coefficients_full = (
        centered @ basis_full
    )

    return PODData(
        parameters=parameters,
        snapshots=snapshots,
        snapshot_matrix=snapshot_matrix,
        reference=reference,
        centered_snapshots=centered,
        basis_full=basis_full,
        singular_values=singular_values,
        projection_coefficients_full=(
            projection_coefficients_full
        ),
    )

# ============================================================
# 4. General steady reduced problem
# ============================================================

@dataclass
class SteadyReducedProblem:
    """
    Description of a problem-specific steady reduced system.

    The neural-network physical output may contain:

        [alpha_1, ..., alpha_m, auxiliary_1, ..., auxiliary_q].

    coefficient_dimension specifies m.

    residual_torch must return a differentiable residual of shape:

        [batch_size, n_reduced_equations].

    In the usual Galerkin case n_reduced_equations = m.
    """

    basis: np.ndarray
    reference: np.ndarray
    input_lower: np.ndarray
    input_upper: np.ndarray
    coefficient_dimension: int
    residual_torch: TorchResidual
    residual_numpy: Optional[NumpyResidual] = None
    name: str = "Steady reduced problem"

    def __post_init__(self) -> None:
        self.basis = np.asarray(
            self.basis,
            dtype=np.float64,
        )

        self.reference = np.asarray(
            self.reference,
            dtype=np.float64,
        ).reshape(-1)

        self.input_lower = np.asarray(
            self.input_lower,
            dtype=np.float64,
        ).reshape(-1)

        self.input_upper = np.asarray(
            self.input_upper,
            dtype=np.float64,
        ).reshape(-1)

        if self.basis.ndim != 2:
            raise ValueError(
                "basis must have shape [full_dimension, m]."
            )

        if self.basis.shape[1] != self.coefficient_dimension:
            raise ValueError(
                "coefficient_dimension must equal "
                "the number of POD basis columns."
            )

        if self.reference.size != self.basis.shape[0]:
            raise ValueError(
                "reference dimension is inconsistent with basis."
            )

        if self.input_lower.shape != self.input_upper.shape:
            raise ValueError(
                "Input lower and upper bounds are inconsistent."
            )

    @property
    def full_dimension(self) -> int:
        return self.basis.shape[0]

    @property
    def parameter_dimension(self) -> int:
        return self.input_lower.size

    def reconstruct(
        self,
        physical_outputs: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct the full field from the first m outputs.
        """
        physical_outputs = np.asarray(
            physical_outputs,
            dtype=np.float64,
        )

        coefficients = physical_outputs[
            :,
            :self.coefficient_dimension,
        ]

        return (
            coefficients @ self.basis.T
            + self.reference[None, :]
        )

    @classmethod
    def from_full_operators(
        cls,
        basis: np.ndarray,
        reference: np.ndarray,
        input_lower: Sequence[float],
        input_upper: Sequence[float],
        linear_operator: Callable[[Tensor], Tensor],
        nonlinear_operator: Callable[[Tensor, Tensor], Tensor],
        constant_operator: Callable[[Tensor], Tensor],
        name: str = "L-g-C reduced problem",
    ) -> "SteadyReducedProblem":
        """
        Convenience constructor for

            V^T [L(mu) state + g(state, mu) + C(mu)].

        Shapes
        ------
        linear_operator(mu):
            [batch, N, N] or [N, N].

        nonlinear_operator(state, mu):
            [batch, N].

        constant_operator(mu):
            [batch, N] or [N].

        This does not parse PDE text. The experiment supplies
        discretized L, g and C.
        """
        basis_numpy = np.asarray(
            basis,
            dtype=np.float64,
        )

        reference_numpy = np.asarray(
            reference,
            dtype=np.float64,
        )

        m = basis_numpy.shape[1]

        def residual_torch(
            outputs: Tensor,
            parameters: Tensor,
        ) -> Tensor:
            basis_tensor = torch.as_tensor(
                basis_numpy,
                dtype=outputs.dtype,
                device=outputs.device,
            )

            reference_tensor = torch.as_tensor(
                reference_numpy,
                dtype=outputs.dtype,
                device=outputs.device,
            )

            coefficients = outputs[:, :m]

            state = (
                coefficients @ basis_tensor.T
                + reference_tensor[None, :]
            )

            linear = linear_operator(parameters)

            if linear.ndim == 2:
                linear_term = state @ linear.T
            elif linear.ndim == 3:
                linear_term = torch.bmm(
                    linear,
                    state[:, :, None],
                ).squeeze(-1)
            else:
                raise ValueError(
                    "linear_operator must return [N,N] "
                    "or [batch,N,N]."
                )

            nonlinear_term = nonlinear_operator(
                state,
                parameters,
            )

            constant_term = constant_operator(parameters)

            if constant_term.ndim == 1:
                constant_term = constant_term[None, :].expand(
                    len(parameters),
                    -1,
                )

            full_residual = (
                linear_term
                + nonlinear_term
                + constant_term
            )

            return full_residual @ basis_tensor

        return cls(
            basis=basis_numpy,
            reference=reference_numpy,
            input_lower=np.asarray(input_lower),
            input_upper=np.asarray(input_upper),
            coefficient_dimension=m,
            residual_torch=residual_torch,
            residual_numpy=None,
            name=name,
        )

# ============================================================
# 5. Configurable feedforward network
# ============================================================

def make_activation(
    specification: ActivationSpecification,
) -> nn.Module:
    if callable(specification) and not isinstance(
        specification,
        str,
    ):
        activation = specification()

        if not isinstance(activation, nn.Module):
            raise TypeError(
                "Activation factory must return nn.Module."
            )

        return activation

    name = str(specification).lower()

    activation_table: Dict[str, Callable[[], nn.Module]] = {
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }

    if name not in activation_table:
        raise ValueError(
            f"Unsupported activation: {specification}. "
            f"Available: {sorted(activation_table)}"
        )

    return activation_table[name]()

class ReducedCoefficientNet(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_width: int = 64,
        hidden_depth: int = 3,
        activation: ActivationSpecification = "silu",
    ) -> None:
        super().__init__()

        if input_dimension <= 0:
            raise ValueError(
                "input_dimension must be positive."
            )

        if output_dimension <= 0:
            raise ValueError(
                "output_dimension must be positive."
            )

        if hidden_width <= 0:
            raise ValueError(
                "hidden_width must be positive."
            )

        if hidden_depth < 0:
            raise ValueError(
                "hidden_depth cannot be negative."
            )

        layers: List[nn.Module] = []
        current_dimension = input_dimension

        for _ in range(hidden_depth):
            layers.append(
                nn.Linear(
                    current_dimension,
                    hidden_width,
                )
            )

            layers.append(
                make_activation(activation)
            )

            current_dimension = hidden_width

        layers.append(
            nn.Linear(
                current_dimension,
                output_dimension,
            )
        )

        self.network = nn.Sequential(*layers)

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.activation_name = (
            activation
            if isinstance(activation, str)
            else activation.__name__
        )

    def forward(
        self,
        normalized_inputs: Tensor,
    ) -> Tensor:
        return self.network(normalized_inputs)

# ============================================================
# 6. Training configuration and results
# ============================================================

@dataclass
class TrainConfig:
    """
    Defaults requested for the general steady model.

    total_epochs = Adam epochs + L-BFGS epochs.
    """

    total_epochs: int = 6000
    lbfgs_epochs: int = 200

    learning_rate: float = 1.0e-2
    learning_rate_gamma: float = 0.95
    learning_rate_step: int = 200

    projection_batches: int = 1
    residual_batches: int = 10

    pdnn_l2_lambda: float = 1.0e-10
    physics_weight: float = 1.0

    hidden_width: int = 64
    hidden_depth: int = 3
    activation: str = "silu"

    lbfgs_lr: float = 1.0
    lbfgs_history_size: int = 50
    lbfgs_tolerance_grad: float = 1.0e-10
    lbfgs_tolerance_change: float = 1.0e-12

    seed: int = 1234
    log_points_per_stage: int = 10

    def __post_init__(self) -> None:
        if self.total_epochs <= 0:
            raise ValueError(
                "total_epochs must be positive."
            )

        if self.lbfgs_epochs < 0:
            raise ValueError(
                "lbfgs_epochs cannot be negative."
            )

        if self.lbfgs_epochs >= self.total_epochs:
            raise ValueError(
                "lbfgs_epochs must be smaller than total_epochs."
            )

        if self.projection_batches <= 0:
            raise ValueError(
                "projection_batches must be positive."
            )

        if self.residual_batches <= 0:
            raise ValueError(
                "residual_batches must be positive."
            )

@dataclass
class TrainResult:
    method: str
    model: ReducedCoefficientNet
    scaling: StandardScaling
    history: List[Dict[str, Any]]
    config: TrainConfig
    best_validation_error: float = math.inf

# ============================================================
# 7. General steady trainer
# ============================================================

class SteadyPRNNTrainer:
    """
    General trainer for PDNN, PINN and PRNN.

    PDNN
    ----
        L = MSE(y_tilde_nn, y_tilde_target)
            + lambda * ||W||_F^2.

    PINN
    ----
        L = MSE(reduced_residual, 0).

    PRNN
    ----
        L = L_data + omega * L_physics.
    """

    def __init__(
        self,
        problem: SteadyReducedProblem,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.problem = problem
        self.device = get_device(device)
        self.dtype = dtype

    def _tensor(
        self,
        values: np.ndarray,
    ) -> Tensor:
        return torch.as_tensor(
            values,
            dtype=self.dtype,
            device=self.device,
        )

    def normalized_inputs(
        self,
        parameters: np.ndarray,
    ) -> np.ndarray:
        return normalize_inputs(
            parameters,
            self.problem.input_lower,
            self.problem.input_upper,
        )

    @staticmethod
    def _weight_frobenius_penalty(
        model: nn.Module,
    ) -> Tensor:
        """
        Sum of squared Linear-layer weights.

        Bias vectors are excluded because the paper writes ||W||_F^2.
        """
        penalty: Optional[Tensor] = None

        for module in model.modules():
            if isinstance(module, nn.Linear):
                term = torch.sum(module.weight ** 2)
                penalty = (
                    term
                    if penalty is None
                    else penalty + term
                )

        if penalty is None:
            raise RuntimeError(
                "The model contains no Linear layer."
            )

        return penalty

    @staticmethod
    def _split_indices(
        n_values: int,
        n_batches: int,
    ) -> List[np.ndarray]:
        if n_values <= 0:
            return []

        indices = np.arange(n_values)
        np.random.shuffle(indices)

        return [
            part
            for part in np.array_split(
                indices,
                min(n_batches, n_values),
            )
            if len(part) > 0
        ]

    def predict_outputs(
        self,
        result: TrainResult,
        parameters: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        parameters = np.asarray(
            parameters,
            dtype=np.float64,
        )

        normalized = self.normalized_inputs(parameters)
        predictions: List[np.ndarray] = []

        result.model.eval()

        with torch.no_grad():
            for start in range(
                0,
                len(normalized),
                batch_size,
            ):
                batch = self._tensor(
                    normalized[start:start + batch_size]
                )

                predictions.append(
                    result.model(batch).cpu().numpy()
                )

        normalized_outputs = np.concatenate(
            predictions,
            axis=0,
        )

        return result.scaling.denormalize(
            normalized_outputs
        )

    def predict_coefficients(
        self,
        result: TrainResult,
        parameters: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        outputs = self.predict_outputs(
            result,
            parameters,
            batch_size=batch_size,
        )

        return outputs[
            :,
            :self.problem.coefficient_dimension,
        ]

    def predict_fields(
        self,
        result: TrainResult,
        parameters: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        outputs = self.predict_outputs(
            result,
            parameters,
            batch_size=batch_size,
        )

        return self.problem.reconstruct(outputs)

    def train(
        self,
        method: str,
        projection_parameters: np.ndarray,
        projection_targets: np.ndarray,
        residual_parameters: np.ndarray,
        config: TrainConfig,
    ) -> TrainResult:
        method = method.upper()

        if method not in {"PDNN", "PINN", "PRNN"}:
            raise ValueError(
                "method must be PDNN, PINN or PRNN."
            )

        set_seed(config.seed)

        projection_parameters = np.asarray(
            projection_parameters,
            dtype=np.float64,
        )

        projection_targets = np.asarray(
            projection_targets,
            dtype=np.float64,
        )

        residual_parameters = np.asarray(
            residual_parameters,
            dtype=np.float64,
        )

        if projection_parameters.ndim != 2:
            raise ValueError(
                "projection_parameters must be two-dimensional."
            )

        if projection_targets.ndim != 2:
            raise ValueError(
                "projection_targets must be two-dimensional."
            )

        if residual_parameters.ndim != 2:
            raise ValueError(
                "residual_parameters must be two-dimensional."
            )

        if len(projection_parameters) != len(
            projection_targets
        ):
            raise ValueError(
                "Projection parameters and targets "
                "must have the same number of rows."
            )

        if projection_parameters.shape[1] != (
            self.problem.parameter_dimension
        ):
            raise ValueError(
                "Projection parameter dimension is inconsistent."
            )

        if residual_parameters.shape[1] != (
            self.problem.parameter_dimension
        ):
            raise ValueError(
                "Residual parameter dimension is inconsistent."
            )

        scaling = StandardScaling.fit(
            projection_targets
        )

        x_projection = self._tensor(
            self.normalized_inputs(
                projection_parameters
            )
        )

        y_projection = self._tensor(
            scaling.normalize(
                projection_targets
            )
        )

        x_residual_normalized = self._tensor(
            self.normalized_inputs(
                residual_parameters
            )
        )

        x_residual_physical = self._tensor(
            residual_parameters
        )

        output_mean = self._tensor(
            scaling.mean
        )

        output_std = self._tensor(
            scaling.std
        )

        model = ReducedCoefficientNet(
            input_dimension=(
                projection_parameters.shape[1]
            ),
            output_dimension=(
                projection_targets.shape[1]
            ),
            hidden_width=config.hidden_width,
            hidden_depth=config.hidden_depth,
            activation=config.activation,
        ).to(
            device=self.device,
            dtype=self.dtype,
        )

        adam_epochs = (
            config.total_epochs
            - config.lbfgs_epochs
        )

        optimizer_adam = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0,
        )

        history: List[Dict[str, Any]] = []

        zero = torch.zeros(
            (),
            dtype=self.dtype,
            device=self.device,
        )

        # ----------------------------------------------------
        # 7.1 PDNN subsection
        # ----------------------------------------------------

        def pdnn_loss(
            projection_indices: Optional[np.ndarray] = None,
            include_regularization: bool = True,
        ) -> Tuple[Tensor, Tensor]:
            if projection_indices is None:
                x_values = x_projection
                y_values = y_projection
            else:
                x_values = x_projection[
                    projection_indices
                ]

                y_values = y_projection[
                    projection_indices
                ]

            prediction = model(x_values)

            data_mse = torch.mean(
                (prediction - y_values) ** 2
            )

            if include_regularization:
                regularization = (
                    config.pdnn_l2_lambda
                    * self._weight_frobenius_penalty(
                        model
                    )
                )
            else:
                regularization = zero

            return (
                data_mse + regularization,
                data_mse,
            )

        # ----------------------------------------------------
        # 7.2 PINN subsection
        # ----------------------------------------------------

        def pinn_loss(
            residual_indices: Optional[np.ndarray] = None,
        ) -> Tensor:
            if residual_indices is None:
                normalized_parameters = (
                    x_residual_normalized
                )

                physical_parameters = (
                    x_residual_physical
                )
            else:
                normalized_parameters = (
                    x_residual_normalized[
                        residual_indices
                    ]
                )

                physical_parameters = (
                    x_residual_physical[
                        residual_indices
                    ]
                )

            normalized_outputs = model(
                normalized_parameters
            )

            physical_outputs = (
                normalized_outputs
                * output_std[None, :]
                + output_mean[None, :]
            )

            residual = self.problem.residual_torch(
                physical_outputs,
                physical_parameters,
            )

            if residual.ndim != 2:
                raise ValueError(
                    "residual_torch must return "
                    "[batch, n_equations]."
                )

            return torch.mean(residual ** 2)

        def complete_loss() -> Tuple[Tensor, Tensor, Tensor]:
            if method == "PDNN":
                total, data = pdnn_loss()
                physics = zero
            elif method == "PINN":
                data = zero
                physics = pinn_loss()
                total = physics
            else:
                data_with_regularization, data_mse = (
                    pdnn_loss()
                )

                physics = pinn_loss()

                total = (
                    data_with_regularization
                    + config.physics_weight
                    * physics
                )

                data = data_mse

            return total, data, physics

        print(
            f"[train] {method:4s} "
            f"Adam={adam_epochs}, "
            f"L-BFGS={config.lbfgs_epochs}, "
            f"width={config.hidden_width}, "
            f"depth={config.hidden_depth}, "
            f"activation={config.activation}, "
            f"omega={config.physics_weight:.2e}",
            flush=True,
        )

        model.train()

        adam_log_step = max(
            1,
            adam_epochs
            // max(1, config.log_points_per_stage),
        )

        for epoch in range(adam_epochs):
            learning_rate_factor = (
                config.learning_rate_gamma
                ** (
                    epoch
                    // config.learning_rate_step
                )
            )

            current_learning_rate = (
                config.learning_rate
                * learning_rate_factor
            )

            for group in optimizer_adam.param_groups:
                group["lr"] = current_learning_rate

            if method == "PDNN":
                projection_batches = self._split_indices(
                    len(projection_parameters),
                    config.projection_batches,
                )

                for projection_indices in projection_batches:
                    optimizer_adam.zero_grad(
                        set_to_none=True
                    )

                    total, _ = pdnn_loss(
                        projection_indices
                    )

                    if not torch.isfinite(total):
                        raise FloatingPointError(
                            "Non-finite PDNN Adam loss."
                        )

                    total.backward()
                    optimizer_adam.step()

            else:
                residual_batches = self._split_indices(
                    len(residual_parameters),
                    config.residual_batches,
                )

                for residual_indices in residual_batches:
                    optimizer_adam.zero_grad(
                        set_to_none=True
                    )

                    physics = pinn_loss(
                        residual_indices
                    )

                    if method == "PINN":
                        total = physics
                    else:
                        data_with_regularization, _ = (
                            pdnn_loss()
                        )

                        total = (
                            data_with_regularization
                            + config.physics_weight
                            * physics
                        )

                    if not torch.isfinite(total):
                        raise FloatingPointError(
                            f"Non-finite {method} Adam loss."
                        )

                    total.backward()
                    optimizer_adam.step()

            if (
                epoch == 0
                or (epoch + 1) % adam_log_step == 0
                or epoch + 1 == adam_epochs
            ):
                total, data, physics = complete_loss()

                row = {
                    "stage": "Adam",
                    "epoch": epoch + 1,
                    "total": float(
                        total.detach().cpu()
                    ),
                    "data": float(
                        data.detach().cpu()
                    ),
                    "physics": float(
                        physics.detach().cpu()
                    ),
                    "learning_rate": (
                        current_learning_rate
                    ),
                }

                history.append(row)

                print(
                    f"    Adam "
                    f"{epoch + 1:5d}/{adam_epochs}: "
                    f"total={row['total']:.4e}, "
                    f"data={row['data']:.4e}, "
                    f"physics={row['physics']:.4e}, "
                    f"lr={row['learning_rate']:.4e}",
                    flush=True,
                )

        if config.lbfgs_epochs > 0:
            optimizer_lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=config.lbfgs_lr,
                max_iter=1,
                history_size=(
                    config.lbfgs_history_size
                ),
                tolerance_grad=(
                    config.lbfgs_tolerance_grad
                ),
                tolerance_change=(
                    config.lbfgs_tolerance_change
                ),
                line_search_fn="strong_wolfe",
            )

            lbfgs_log_step = max(
                1,
                config.lbfgs_epochs
                // max(
                    1,
                    config.log_points_per_stage,
                ),
            )

            for step in range(
                config.lbfgs_epochs
            ):
                cache: Dict[str, Tensor] = {}

                def closure() -> Tensor:
                    optimizer_lbfgs.zero_grad(
                        set_to_none=True
                    )

                    total, data, physics = (
                        complete_loss()
                    )

                    if not torch.isfinite(total):
                        raise FloatingPointError(
                            f"Non-finite {method} "
                            "L-BFGS loss."
                        )

                    total.backward()

                    cache["total"] = total.detach()
                    cache["data"] = data.detach()
                    cache["physics"] = (
                        physics.detach()
                    )

                    return total

                optimizer_lbfgs.step(closure)

                if (
                    step == 0
                    or (
                        step + 1
                    ) % lbfgs_log_step == 0
                    or (
                        step + 1
                        == config.lbfgs_epochs
                    )
                ):
                    total, data, physics = (
                        complete_loss()
                    )

                    row = {
                        "stage": "LBFGS",
                        "epoch": step + 1,
                        "total": float(
                            total.detach().cpu()
                        ),
                        "data": float(
                            data.detach().cpu()
                        ),
                        "physics": float(
                            physics.detach().cpu()
                        ),
                    }

                    history.append(row)

                    print(
                        f"    LBFGS "
                        f"{step + 1:5d}/"
                        f"{config.lbfgs_epochs}: "
                        f"total={row['total']:.4e}, "
                        f"data={row['data']:.4e}, "
                        f"physics={row['physics']:.4e}",
                        flush=True,
                    )

        return TrainResult(
            method=method,
            model=model,
            scaling=scaling,
            history=history,
            config=copy.deepcopy(config),
        )

# ============================================================
# 8. Checkpoint support
# ============================================================

def save_checkpoint(
    path: Union[Path, str],
    result: TrainResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "method": result.method,
            "state_dict": (
                result.model.state_dict()
            ),
            "input_dimension": (
                result.model.input_dimension
            ),
            "output_dimension": (
                result.model.output_dimension
            ),
            "hidden_width": (
                result.model.hidden_width
            ),
            "hidden_depth": (
                result.model.hidden_depth
            ),
            "activation": (
                result.model.activation_name
            ),
            "scaling_mean": (
                result.scaling.mean
            ),
            "scaling_std": (
                result.scaling.std
            ),
            "history": result.history,
            "config": asdict(result.config),
            "best_validation_error": (
                result.best_validation_error
            ),
            "metadata": metadata or {},
        },
        path,
    )

def load_checkpoint(
    path: Union[Path, str],
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> TrainResult:
    path = Path(path)
    device_object = get_device(device)

    data = torch.load(
        path,
        map_location=device_object,
        weights_only=False,
    )

    activation = data.get(
        "activation",
        "silu",
    )

    model = ReducedCoefficientNet(
        input_dimension=int(
            data["input_dimension"]
        ),
        output_dimension=int(
            data["output_dimension"]
        ),
        hidden_width=int(
            data["hidden_width"]
        ),
        hidden_depth=int(
            data["hidden_depth"]
        ),
        activation=activation,
    ).to(
        device=device_object,
        dtype=dtype,
    )

    model.load_state_dict(
        data["state_dict"]
    )

    config_dictionary = data.get(
        "config",
        {},
    )

    config = TrainConfig(
        **{
            key: value
            for key, value
            in config_dictionary.items()
            if key in TrainConfig.__dataclass_fields__
        }
    )

    return TrainResult(
        method=str(data["method"]),
        model=model,
        scaling=StandardScaling(
            mean=np.asarray(
                data["scaling_mean"],
                dtype=np.float64,
            ),
            std=np.asarray(
                data["scaling_std"],
                dtype=np.float64,
            ),
        ),
        history=data.get(
            "history",
            [],
        ),
        config=config,
        best_validation_error=float(
            data.get(
                "best_validation_error",
                math.inf,
            )
        ),
    )

def save_training_history(
    path: Union[Path, str],
    result: TrainResult,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "method": result.method,
                "best_validation_error": (
                    result.best_validation_error
                ),
                "config": asdict(result.config),
                "history": result.history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
