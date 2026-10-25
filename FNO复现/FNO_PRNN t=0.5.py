import os
import torch
from torch import nn
from torch.func import jvp
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
batch_size = 20
epochs = 500
width = 64
modes = 16
rank = 36
physics_weight = 0.1
x = torch.arange(n) * dx
beta = float(4 * nu * torch.sin(torch.tensor(torch.pi / n))**2 / dx**2)


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


class FNO_PRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = nn.Conv1d(2, width, 1)
        self.weights = nn.ParameterList()
        for i in range(4):
            weight = torch.randn(width, width, modes, dtype=torch.cfloat) / width
            self.weights.append(nn.Parameter(weight))
        self.local = nn.ModuleList()
        for i in range(4):
            self.local.append(nn.Conv1d(width, width, 1))
        self.project = nn.Sequential(nn.Conv1d(width, 128, 1), nn.ReLU(), nn.Conv1d(128, 1, 1))

    def forward(self, u, t):
        v = self.lift(torch.stack((u, t.expand_as(u)), dim=1))
        for i in range(4):
            spectrum = torch.fft.rfft(v)
            low = torch.einsum("bik,iok->bok", spectrum[:, :, :modes], self.weights[i])
            high = torch.zeros_like(spectrum[:, :, modes:])
            out = torch.cat((low, high), dim=-1)
            v = torch.nn.functional.relu(torch.fft.irfft(out, n=n) + self.local[i](v))
        correction = self.project(v)[:, 0, :] @ basis / n
        initial = u @ basis / n
        decay = torch.exp(-beta * t)
        return initial * torch.exp(t * rates) + (1 - decay) * decay * correction


k = torch.arange(1, n // 2)[:, None]
phase = 2 * torch.pi * k * x
amplitude = 2**0.5 * 25 / ((2 * torch.pi * k[:, 0])**2 + 25)
a = torch.randn(train_size + test_size, len(k)) * amplitude
b = torch.randn(train_size + test_size, len(k)) * amplitude
fields = a @ phase.sin() + b @ phase.cos()
inputs = fields[:train_size]
test_inputs = fields[train_size:]
targets, snapshots = solve(inputs)
scale = targets.std()
initial_scale = inputs.square().mean().sqrt()
basis, rates, quadratic = make_pod(snapshots)
labels = targets @ basis / n

test_targets, _ = solve(test_inputs)
inputs = inputs.to(device)
labels = labels.to(device)
test_inputs = test_inputs.to(device)
test_targets = test_targets.to(device)
basis = basis.to(device)
rates = rates.to(device)
quadratic = quadratic.to(device)
initial_scale = initial_scale.to(device)

model = FNO_PRNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
time_generator = torch.Generator().manual_seed(1)

history = []

for epoch in range(epochs):
    total = torch.zeros(3, device=device)
    for ids in torch.randperm(len(inputs)).to(device).split(batch_size):
        u = inputs[ids]
        prediction = model(u, torch.ones(len(ids), 1, device=device) * T)
        difference = torch.linalg.vector_norm(prediction - labels[ids], dim=-1)
        norm = torch.linalg.vector_norm(labels[ids], dim=-1)
        data_loss = (difference / norm.clamp_min(1e-12)).mean()
        t = torch.rand(len(ids), 1, generator=time_generator).to(device) * T
        coefficients, derivative = jvp(model, (u, t), (torch.zeros_like(u), torch.ones_like(t)))
        convection = torch.einsum("ijk,bj,bk->bi", quadratic, coefficients, coefficients)
        residual = derivative - (coefficients * rates + convection)
        residual_scale = beta * initial_scale * torch.exp(-beta * t)
        physics_loss = (residual / residual_scale).square().sum(-1).mean()
        loss = data_loss + physics_weight * physics_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += torch.stack((loss.detach(), data_loss.detach(), physics_weight * physics_loss.detach())) * len(ids)
    schedule.step()
    total = total / len(inputs)
    history.append([epoch + 1, total[0].item(), total[1].item(), total[2].item()])
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: total={total[0]:.8f}, "
              f"data={total[1]:.8f}, weighted_physics={total[2]:.8f}")

model.eval()
with torch.no_grad():
    test_prediction = torch.cat([model(u, torch.ones(len(u), 1, device=device) * T) @ basis.T
                                 for u in test_inputs.split(batch_size)])
test_error = torch.linalg.vector_norm(test_prediction - test_targets, dim=-1)
test_error = test_error / torch.linalg.vector_norm(test_targets, dim=-1)
print(f"Mean test relative L2 error ({test_size} samples): {test_error.mean().item():.6%}")

u0 = test_inputs[0]
reference = test_targets[0]
model.eval()
with torch.no_grad():
    prediction = model(u0.unsqueeze(0), torch.ones(1, 1, device=device) * T) @ basis.T
    prediction = prediction[0]

u0, reference, prediction = u0.cpu(), reference.cpu(), prediction.cpu()
error = torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)
print(f"Relative L2 error: {error.item():.6%}")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
history = torch.tensor(history)
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)

axes[0, 0].plot(x, u0, "k-")
axes[0, 0].set_title("(a) 固定测试初始场")
axes[0, 0].set_ylabel(r"$u_0(x)$")

axes[0, 1].plot(x, reference, "k-", label="数值参考解")
axes[0, 1].plot(x, prediction, "k--", label="FNO-PRNN 预测")
axes[0, 1].set_title(f"(b) t = {T:g} 时的解")
axes[0, 1].set_ylabel(f"$u(x, {T:g})$")
axes[0, 1].legend(frameon=False)

axes[1, 0].plot(x, (prediction - reference).abs(), "k-")
axes[1, 0].set_title(f"(c) t = {T:g} 时的绝对误差")
axes[1, 0].set_ylabel(r"$|\hat{u}-u_{\mathrm{ref}}|$")
axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[1, 1].semilogy(history[:, 0], history[:, 1], "k-", label="加权总损失")
axes[1, 1].semilogy(history[:, 0], history[:, 2], "k--", label="数据损失")
axes[1, 1].semilogy(history[:, 0], history[:, 3], "k:", label=f"物理损失 × {physics_weight:g}")
axes[1, 1].set_title("(d) 每轮记录的训练损失")
axes[1, 1].set_ylabel("归一化损失")
axes[1, 1].legend(frameon=False)
axes[1, 1].set_xlabel("训练轮次")

for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set_xlabel("x")
    ax.set_xlim(0, 1)

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.show()
