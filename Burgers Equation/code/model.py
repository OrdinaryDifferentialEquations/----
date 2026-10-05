#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor
ResidualFunction = Callable[[Tensor, Tensor], Tensor]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_inputs(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    return 2.0 * (x - lower) / (upper - lower) - 1.0

@dataclass
class OutputScaling:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "OutputScaling":
        values = np.asarray(values, dtype=np.float64)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std = np.maximum(std, 1.0e-12)
        return cls(mean=mean, std=std)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

@dataclass
class TrainConfig:
    epochs: int = 20000
    adam_fraction: float = 0.90
    learning_rate: float = 1.0e-2
    learning_rate_gamma: float = 0.96
    learning_rate_step: int = 200
    residual_batches: int = 10
    projection_batches: int = 1
    pdnn_weight_decay: float = 1.0e-4
    physics_weight: float = 1.0
    seed: int = 1234
    lbfgs_lr: float = 1.0
    lbfgs_history_size: int = 50
    lbfgs_tolerance_grad: float = 1.0e-10
    lbfgs_tolerance_change: float = 1.0e-12
    log_every: int = 200

@dataclass
class TrainResult:
    method: str
    model: "ReducedCoefficientNet"
    scaling: OutputScaling
    history: List[Dict[str, float]]
    best_validation_error: float = np.inf

class ReducedCoefficientNet(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_width: int = 20,
        hidden_depth: int = 3,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        current_dimension = input_dimension

        for _ in range(hidden_depth):
            layers.append(nn.Linear(current_dimension, hidden_width))
            layers.append(nn.SiLU())
            current_dimension = hidden_width

        layers.append(nn.Linear(current_dimension, output_dimension))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

class ReducedPDE:
    def __init__(
        self,
        basis: np.ndarray,
        input_lower: Sequence[float],
        input_upper: Sequence[float],
        residual_function: Optional[ResidualFunction] = None,
        linear_operator: Optional[
            Callable[[Tensor], Tensor] | Tensor | np.ndarray
        ] = None,
        nonlinear_operator: Optional[
            Callable[[Tensor, Tensor], Tensor]
        ] = None,
        constant_operator: Optional[
            Callable[[Tensor], Tensor] | Tensor | np.ndarray
        ] = None,
        reference: Optional[np.ndarray] = None,
    ) -> None:
        basis = np.asarray(basis, dtype=np.float64)

        if basis.ndim != 2:
            raise ValueError("basis must have shape [full_dimension, m].")

        self.basis_numpy = basis
        self.input_lower = np.asarray(input_lower, dtype=np.float64)
        self.input_upper = np.asarray(input_upper, dtype=np.float64)

        if reference is None:
            reference = np.zeros(basis.shape[0], dtype=np.float64)

        self.reference_numpy = np.asarray(reference, dtype=np.float64)
        self.residual_function = residual_function
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.constant_operator = constant_operator

    def _operator_value(
        self,
        operator,
        parameters: Tensor,
        full_dimension: int,
    ) -> Tensor:
        if operator is None:
            return torch.zeros(
                (parameters.shape[0], full_dimension),
                dtype=parameters.dtype,
                device=parameters.device,
            )

        if callable(operator):
            return operator(parameters)

        value = torch.as_tensor(
            operator,
            dtype=parameters.dtype,
            device=parameters.device,
        )

        if value.ndim == 1:
            return value[None, :].expand(parameters.shape[0], -1)

        return value

    def residual(
        self,
        coefficients: Tensor,
        parameters: Tensor,
    ) -> Tensor:
        if self.residual_function is not None:
            return self.residual_function(coefficients, parameters)

        basis = torch.as_tensor(
            self.basis_numpy,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )

        reference = torch.as_tensor(
            self.reference_numpy,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )

        state = coefficients @ basis.T + reference[None, :]
        batch_size = coefficients.shape[0]
        full_dimension = basis.shape[0]

        linear = self.linear_operator

        if callable(linear):
            linear_matrix = linear(parameters)
        else:
            linear_matrix = torch.as_tensor(
                linear,
                dtype=coefficients.dtype,
                device=coefficients.device,
            )

        if linear_matrix.ndim == 2:
            linear_term = state @ linear_matrix.T
        elif linear_matrix.ndim == 3:
            linear_term = torch.bmm(
                linear_matrix,
                state[:, :, None],
            ).squeeze(-1)
        else:
            raise ValueError(
                "linear_operator must return [N,N] or [B,N,N]."
            )

        if self.nonlinear_operator is None:
            nonlinear_term = torch.zeros_like(linear_term)
        else:
            nonlinear_term = self.nonlinear_operator(state, parameters)

        constant_term = self._operator_value(
            self.constant_operator,
            parameters,
            full_dimension,
        )

        full_residual = linear_term + nonlinear_term + constant_term
        return full_residual @ basis

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        return (
            coefficients @ self.basis_numpy.T
            + self.reference_numpy[None, :]
        )

class PRNNTrainer:
    def __init__(
        self,
        reduced_pde: ReducedPDE,
        hidden_width: int = 20,
        hidden_depth: int = 3,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.reduced_pde = reduced_pde
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.device = get_device(device)
        self.dtype = dtype

    def _tensor(self, values: np.ndarray) -> Tensor:
        return torch.as_tensor(
            values,
            dtype=self.dtype,
            device=self.device,
        )

    def normalized_input(self, parameters: np.ndarray) -> np.ndarray:
        return normalize_inputs(
            parameters,
            self.reduced_pde.input_lower,
            self.reduced_pde.input_upper,
        )

    def predict(
        self,
        model: ReducedCoefficientNet,
        scaling: OutputScaling,
        parameters: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        normalized = self.normalized_input(parameters)
        outputs: List[np.ndarray] = []

        model.eval()

        with torch.no_grad():
            for start in range(0, len(normalized), batch_size):
                values = self._tensor(
                    normalized[start:start + batch_size]
                )
                outputs.append(model(values).cpu().numpy())

        normalized_coefficients = np.concatenate(outputs, axis=0)
        return scaling.denormalize(normalized_coefficients)

    def train(
        self,
        method: str,
        projection_parameters: np.ndarray,
        projection_coefficients: np.ndarray,
        residual_parameters: np.ndarray,
        config: TrainConfig,
    ) -> TrainResult:
        method = method.upper()

        if method not in {"PDNN", "PINN", "PRNN"}:
            raise ValueError("method must be PDNN, PINN, or PRNN.")

        set_seed(config.seed)

        projection_parameters = np.asarray(
            projection_parameters,
            dtype=np.float64,
        )
        projection_coefficients = np.asarray(
            projection_coefficients,
            dtype=np.float64,
        )
        residual_parameters = np.asarray(
            residual_parameters,
            dtype=np.float64,
        )

        scaling = OutputScaling.fit(projection_coefficients)

        x_projection = self._tensor(
            self.normalized_input(projection_parameters)
        )
        y_projection = self._tensor(
            scaling.normalize(projection_coefficients)
        )
        x_residual_normalized = self._tensor(
            self.normalized_input(residual_parameters)
        )
        x_residual_physical = self._tensor(residual_parameters)

        coefficient_mean = self._tensor(scaling.mean)
        coefficient_std = self._tensor(scaling.std)

        model = ReducedCoefficientNet(
            input_dimension=projection_parameters.shape[1],
            output_dimension=projection_coefficients.shape[1],
            hidden_width=self.hidden_width,
            hidden_depth=self.hidden_depth,
        ).to(device=self.device, dtype=self.dtype)

        weight_decay = (
            config.pdnn_weight_decay if method == "PDNN" else 0.0
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=weight_decay,
        )

        n_adam = int(config.epochs * config.adam_fraction)
        n_lbfgs = config.epochs - n_adam
        history: List[Dict[str, float]] = []

        def data_loss() -> Tensor:
            prediction = model(x_projection)
            return torch.mean((prediction - y_projection) ** 2)

        def physics_loss(
            normalized_parameters: Tensor,
            physical_parameters: Tensor,
        ) -> Tensor:
            normalized_coefficients = model(normalized_parameters)
            coefficients = (
                normalized_coefficients * coefficient_std[None, :]
                + coefficient_mean[None, :]
            )
            residual = self.reduced_pde.residual(
                coefficients,
                physical_parameters,
            )
            return torch.mean(residual ** 2)

        def full_loss() -> Tuple[Tensor, Tensor, Tensor]:
            zero = torch.zeros(
                (),
                dtype=self.dtype,
                device=self.device,
            )

            loss_data = data_loss() if method != "PINN" else zero

            loss_physics = (
                physics_loss(
                    x_residual_normalized,
                    x_residual_physical,
                )
                if method != "PDNN"
                else zero
            )

            if method == "PDNN":
                total = loss_data
            elif method == "PINN":
                total = loss_physics
            else:
                total = (
                    loss_data
                    + config.physics_weight * loss_physics
                )

            return total, loss_data, loss_physics

        residual_indices = np.arange(len(residual_parameters))
        residual_batches = max(1, config.residual_batches)

        for epoch in range(n_adam):
            learning_rate_factor = (
                config.learning_rate_gamma
                ** (epoch // config.learning_rate_step)
            )

            for group in optimizer.param_groups:
                group["lr"] = (
                    config.learning_rate * learning_rate_factor
                )

            if method == "PDNN":
                optimizer.zero_grad(set_to_none=True)
                total, loss_data, loss_physics = full_loss()
                total.backward()
                optimizer.step()
            else:
                np.random.shuffle(residual_indices)
                split_indices = np.array_split(
                    residual_indices,
                    residual_batches,
                )

                for batch_indices in split_indices:
                    optimizer.zero_grad(set_to_none=True)

                    batch_normalized = (
                        x_residual_normalized[batch_indices]
                    )
                    batch_physical = (
                        x_residual_physical[batch_indices]
                    )

                    loss_physics = physics_loss(
                        batch_normalized,
                        batch_physical,
                    )

                    if method == "PINN":
                        loss_data = torch.zeros_like(loss_physics)
                        total = loss_physics
                    else:
                        loss_data = data_loss()
                        total = (
                            loss_data
                            + config.physics_weight * loss_physics
                        )

                    total.backward()
                    optimizer.step()

            if (
                epoch == 0
                or (epoch + 1) % config.log_every == 0
                or epoch + 1 == n_adam
            ):
                total, loss_data, loss_physics = full_loss()

                history.append(
                    {
                        "stage": 0.0,
                        "epoch": float(epoch + 1),
                        "total": float(total.detach().cpu()),
                        "data": float(loss_data.detach().cpu()),
                        "physics": float(
                            loss_physics.detach().cpu()
                        ),
                    }
                )

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lbfgs_lr,
            max_iter=1,
            history_size=config.lbfgs_history_size,
            tolerance_grad=config.lbfgs_tolerance_grad,
            tolerance_change=config.lbfgs_tolerance_change,
            line_search_fn="strong_wolfe",
        )

        for step in range(n_lbfgs):
            cache: Dict[str, Tensor] = {}

            def closure() -> Tensor:
                lbfgs.zero_grad(set_to_none=True)
                total, loss_data, loss_physics = full_loss()
                total.backward()

                cache["total"] = total.detach()
                cache["data"] = loss_data.detach()
                cache["physics"] = loss_physics.detach()
                return total

            lbfgs.step(closure)

            if (
                step == 0
                or (step + 1) % config.log_every == 0
                or step + 1 == n_lbfgs
            ):
                total, loss_data, loss_physics = full_loss()

                history.append(
                    {
                        "stage": 1.0,
                        "epoch": float(step + 1),
                        "total": float(total.detach().cpu()),
                        "data": float(loss_data.detach().cpu()),
                        "physics": float(
                            loss_physics.detach().cpu()
                        ),
                    }
                )

        return TrainResult(
            method=method,
            model=model,
            scaling=scaling,
            history=history,
        )

def save_checkpoint(
    path: Path | str,
    result: TrainResult,
    metadata: Optional[Dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "method": result.method,
            "state_dict": result.model.state_dict(),
            "input_dimension": (
                result.model.network[0].in_features
            ),
            "output_dimension": (
                result.model.network[-1].out_features
            ),
            "scaling_mean": result.scaling.mean,
            "scaling_std": result.scaling.std,
            "history": result.history,
            "best_validation_error": (
                result.best_validation_error
            ),
            "metadata": metadata or {},
        },
        path,
    )

def load_checkpoint(
    path: Path | str,
    hidden_width: int = 20,
    hidden_depth: int = 3,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> TrainResult:
    device_object = get_device(device)

    data = torch.load(
        Path(path),
        map_location=device_object,
        weights_only=False,
    )

    model = ReducedCoefficientNet(
        input_dimension=int(data["input_dimension"]),
        output_dimension=int(data["output_dimension"]),
        hidden_width=hidden_width,
        hidden_depth=hidden_depth,
    ).to(device=device_object, dtype=dtype)

    model.load_state_dict(data["state_dict"])

    return TrainResult(
        method=data["method"],
        model=model,
        scaling=OutputScaling(
            mean=np.asarray(data["scaling_mean"]),
            std=np.asarray(data["scaling_std"]),
        ),
        history=data.get("history", []),
        best_validation_error=float(
            data.get("best_validation_error", np.inf)
        ),
    )
