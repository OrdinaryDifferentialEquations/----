import os
import torch
from torch import nn
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
T = 0.75
train_size = 1000
test_size = 200
batch_size = 20
epochs = 500
width = 64
modes = 16
x = torch.arange(n) * dx

def solve(u):
    u = u.clone()
    for _ in range(round(T / dt)):
        left = torch.roll(u, 1, -1)
        right = torch.roll(u, -1, -1)
        u = u + dt * (-(right**2 - left**2) / (4 * dx)
                      + nu * (right - 2 * u + left) / dx**2)
    return u


class FNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = nn.Conv1d(1, width, 1)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)
            for _ in range(4)
        ])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
        self.project = nn.Sequential(nn.Conv1d(width, 128, 1), nn.ReLU(),
                                     nn.Conv1d(128, 1, 1))

    def forward(self, u):
        v = self.lift(u[:, None, :])
        for weight, local in zip(self.weights, self.local):
            spectrum = torch.fft.rfft(v)
            out = torch.zeros_like(spectrum)
            out[:, :, :modes] = torch.einsum("bik,iok->bok", spectrum[:, :, :modes], weight)
            v = torch.nn.functional.relu(torch.fft.irfft(out, n=v.shape[-1]) + local(v))
        return self.project(v)[:, 0, :]


k = torch.arange(1, n // 2)[:, None]
phase = 2 * torch.pi * k * x
amplitude = 2**0.5 * 25 / ((2 * torch.pi * k[:, 0])**2 + 25)
a = torch.randn(train_size + test_size, len(k)) * amplitude
b = torch.randn(train_size + test_size, len(k)) * amplitude
fields = a @ phase.sin() + b @ phase.cos()
inputs = fields[:train_size]
test_inputs = fields[train_size:]
targets = solve(inputs)
scale = targets.std()
targets = targets / scale

test_targets = solve(test_inputs)
inputs = inputs.to(device)
targets = targets.to(device)
test_inputs = test_inputs.to(device)
test_targets = test_targets.to(device)
scale = scale.to(device)

model = FNO().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

history = []

for epoch in range(epochs):
    for ids in torch.randperm(len(inputs)).to(device).split(batch_size):
        prediction = model(inputs[ids])
        difference = torch.linalg.vector_norm(prediction - targets[ids], dim=-1)
        norm = torch.linalg.vector_norm(targets[ids], dim=-1)
        loss = (difference / norm.clamp_min(1e-12)).mean()
        mse = (prediction - targets[ids]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    schedule.step()
    if (epoch + 1) % 50 == 0:
        history.append([epoch + 1, mse.item()])
        print(f"Epoch {epoch + 1}: relative L2 = {loss.item():.6f}, MSE = {mse.item():.6f}")

model.eval()
with torch.no_grad():
    test_prediction = torch.cat([model(u) * scale for u in test_inputs.split(batch_size)])
test_error = torch.linalg.vector_norm(test_prediction - test_targets, dim=-1)
test_error = test_error / torch.linalg.vector_norm(test_targets, dim=-1)
print(f"Mean test relative L2 error ({test_size} samples): {test_error.mean().item():.6%}")

u0 = test_inputs[0]
reference = test_targets[0]
model.eval()
with torch.no_grad():
    prediction = model(u0[None, :])[0] * scale

u0, reference, prediction = u0.cpu(), reference.cpu(), prediction.cpu()
error = torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)
print(f"Relative L2 error: {error.item():.2%}")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
history = torch.tensor(history)
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)

axes[0, 0].plot(x, u0, "k-")
axes[0, 0].set_title("(a) 固定测试初始场")
axes[0, 0].set_ylabel(r"$u_0(x)$")

axes[0, 1].plot(x, reference, "k-", label="数值参考解")
axes[0, 1].plot(x, prediction, "k--", label="FNO 预测")
axes[0, 1].set_title(f"(b) t = {T:g} 时的解")
axes[0, 1].set_ylabel(f"$u(x, {T:g})$")
axes[0, 1].legend(frameon=False)

axes[1, 0].plot(x, (prediction - reference).abs(), "k-")
axes[1, 0].set_title(f"(c) t = {T:g} 时的绝对误差")
axes[1, 0].set_ylabel(r"$|\hat{u}-u_{\mathrm{ref}}|$")
axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[1, 1].semilogy(history[:, 0], history[:, 1], "k-o", markersize=3)
axes[1, 1].set_title("(d) 记录轮次最后一批的训练损失")
axes[1, 1].set_ylabel("归一化均方误差")
axes[1, 1].set_xlabel("训练轮次")

for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set_xlabel("x")
    ax.set_xlim(0, 1)

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.show()
