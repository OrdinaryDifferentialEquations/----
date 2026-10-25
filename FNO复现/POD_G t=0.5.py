import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.set_num_threads(min(4, os.cpu_count() or 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    print("计算设备：", torch.cuda.get_device_name(0))
else:
    print(f"计算设备：CPU，使用 {torch.get_num_threads()} 个线程（当前 PyTorch 未启用 CUDA）")

nu = 0.1
n = 256
dx = 1 / n
dt = 0.00005
T = 0.5
train_size = 1000
test_size = 200
rank = 12
x = torch.arange(n) * dx


def solve(u):
    u = u.clone()
    snapshots = [u.clone()]
    for step in range(round(T / dt)):
        left = torch.roll(u, 1, -1)
        right = torch.roll(u, -1, -1)
        u = u + dt * (-(right**2 - left**2) / (4 * dx)
                      + nu * (right - 2 * u + left) / dx**2)
        if (step + 1) % 1000 == 0:
            snapshots.append(u.clone())
    return u, torch.cat(snapshots)


def make_pod(snapshots):
    snapshots = snapshots.double()
    snapshots = snapshots - snapshots.mean(-1, keepdim=True)
    covariance = snapshots.T @ snapshots / len(snapshots)
    average = torch.zeros_like(covariance)
    for j in range(n):
        average = average + torch.roll(covariance, (j, j), (0, 1))
    energy, vectors = torch.linalg.eigh(average / n)
    basis = vectors[:, -rank:]
    basis = basis - basis.mean(0, keepdim=True)
    basis = torch.linalg.qr(basis).Q * n**0.5
    laplacian = (torch.roll(basis, -1, 0) - 2 * basis + torch.roll(basis, 1, 0)) / dx**2
    diffusion = nu * basis.T @ laplacian / n
    rates, rotation = torch.linalg.eigh((diffusion + diffusion.T) / 2)
    basis = basis @ rotation
    pairs = basis.T.unsqueeze(1) * basis.T.unsqueeze(0)
    convection = -(torch.roll(pairs, -1, -1) - torch.roll(pairs, 1, -1)) / (4 * dx)
    quadratic = torch.einsum("xi,jkx->ijk", basis, convection) / n
    return basis.float(), rates.float(), quadratic.float()


def solve_pod(u):
    coefficients = u.double() @ basis.double() / n
    linear = rates.double()
    tensor = quadratic.double()
    symmetric = tensor + tensor.transpose(1, 2)
    diagonal = torch.diag(1 - dt * linear)
    times = [0.0]
    energy = [float(coefficients[0].square().sum() / 2)]
    max_residual = 0.0
    for step in range(round(T / dt)):
        previous = coefficients.clone()
        for iteration in range(10):
            rhs = coefficients * linear + torch.einsum("ijk,bj,bk->bi", tensor, coefficients, coefficients)
            residual = coefficients - previous - dt * rhs
            relative = residual.norm(dim=-1) / previous.norm(dim=-1).clamp_min(1e-12)
            if not torch.isfinite(relative).all():
                raise RuntimeError("POD-G 积分出现非有限数值。")
            if relative.max() < 1e-10:
                max_residual = max(max_residual, float(relative.max()))
                break
            jacobian = diagonal - dt * torch.einsum("ijk,bk->bij", symmetric, coefficients)
            change = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
            coefficients = coefficients - change
        else:
            raise RuntimeError("POD-G 后向 Euler 的 Newton 迭代未收敛。")
        if (step + 1) % 1000 == 0:
            times.append((step + 1) * dt)
            energy.append(float(coefficients[0].square().sum() / 2))
            print(f"POD-G 已积分至 t = {(step + 1) * dt:g}", flush=True)
    print(f"POD-G 最大相对时间步方程残差：{max_residual:.3e}")
    return coefficients @ basis.double().T, times, energy


k = torch.arange(1, n // 2)[:, None]
phase = 2 * torch.pi * k * x
amplitude = 2**0.5 * 25 / ((2 * torch.pi * k[:, 0])**2 + 25)
a = torch.randn(train_size + test_size, len(k)) * amplitude
b = torch.randn(train_size + test_size, len(k)) * amplitude
fields = a @ phase.sin() + b @ phase.cos()
inputs = fields[:train_size]
test_inputs = fields[train_size:]
targets, snapshots = solve(inputs)
basis, rates, quadratic = make_pod(snapshots)
del snapshots
test_targets, test_snapshots = solve(test_inputs)
reference_energy = test_snapshots.reshape(-1, test_size, n)[:, 0].square().mean(-1) / 2
del test_snapshots
test_inputs = test_inputs.to(device)
test_targets = test_targets.to(device)
basis = basis.to(device)
rates = rates.to(device)
quadratic = quadratic.to(device)

print("POD-G 使用原 1000 条训练轨迹构基；无神经网络训练或训练轮次。")
with torch.no_grad():
    test_prediction, times, energy = solve_pod(test_inputs)
test_error = torch.linalg.vector_norm(test_prediction - test_targets, dim=-1)
test_error = test_error / torch.linalg.vector_norm(test_targets, dim=-1)
print(f"Mean test relative L2 error ({test_size} samples): {test_error.mean().item():.6%}")

u0 = test_inputs[0].cpu()
reference = test_targets[0].cpu()
prediction = test_prediction[0].cpu()
error = torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)
print(f"Relative L2 error: {error.item():.6%}")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)

axes[0, 0].plot(x, u0, "k-")
axes[0, 0].set_title("(a) 固定测试初始场")
axes[0, 0].set_ylabel(r"$u_0(x)$")

axes[0, 1].plot(x, reference, "k-", label="数值参考解")
axes[0, 1].plot(x, prediction, "k--", label="POD-G 预测")
axes[0, 1].set_title(f"(b) t = {T:g} 时的解")
axes[0, 1].set_ylabel(f"$u(x, {T:g})$")
axes[0, 1].legend(frameon=False)

axes[1, 0].plot(x, (prediction - reference).abs(), "k-")
axes[1, 0].set_title(f"(c) t = {T:g} 时的绝对误差")
axes[1, 0].set_ylabel(r"$|\hat{u}-u_{\mathrm{ref}}|$")
axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[1, 1].semilogy(times, reference_energy, "k-o", markersize=3, label="数值参考解")
axes[1, 1].semilogy(times, energy, "k--s", markersize=3, label="POD-G")
axes[1, 1].set_title("(d) 固定测试样本的能量演化")
axes[1, 1].set_ylabel(r"$E(t)=\frac{1}{2N}\sum_j u_j(t)^2$")
axes[1, 1].legend(frameon=False)
axes[1, 1].set_xlabel("时间 t")

for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set_xlabel("x")
    ax.set_xlim(0, 1)

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

output = Path(__file__).resolve().parent / "结果"
output.mkdir(exist_ok=True)
fig.savefig(output / f"POD-G t={T:g}_四格图.png", dpi=200)
plt.show()
