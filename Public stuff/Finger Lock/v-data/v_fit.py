import numpy as np
import matplotlib.pyplot as plt

# 1. Load Anna's V data and scale x so it's between about -2 and 2
x, y = np.loadtxt("annacode-v.tsv", skiprows=1, usecols=(1, 2)).T
xs = (x - 60) / 20

# 2. Hyperparameters
learning_rate = 0.2
epochs = 300

# ---------------- Model 1: one neuron + ReLU (same as the Excel) ----------------
m, b = 1.0, 1.0
loss1 = []
for epoch in range(epochs):
    z = m * xs + b                  # forward pass
    a = np.maximum(0, z)            # ReLU
    relu_slope = (z >= 0) * 1.0     # ReLU'(z): 1 if on, 0 if off
    error = y - a
    loss1.append(np.mean(error**2))

    dm = np.mean(-2 * error * relu_slope * xs)   # chain rule
    db = np.mean(-2 * error * relu_slope)
    m -= learning_rate * dm
    b -= learning_rate * db

print(f"One neuron:  m = {m:.3f}, b = {b:.2f}, loss = {loss1[-1]:.1f}")

# ---------------- Model 2: two ReLU neurons + an output neuron -----------------
# y is scaled down by 40 too, so the bigger network trains smoothly.
ys = y / 40
w1, c1 = 1.0, 0.0    # neuron 1 starts sloping up
w2, c2 = -1.0, 0.0   # neuron 2 starts sloping down
v1, v2, d = 0.5, 0.5, 0.0   # output neuron: y = v1*h1 + v2*h2 + d
lr = 0.1
loss2 = []
for epoch in range(epochs):
    # forward pass
    z1, z2 = w1 * xs + c1, w2 * xs + c2
    h1, h2 = np.maximum(0, z1), np.maximum(0, z2)
    pred = v1 * h1 + v2 * h2 + d
    error = ys - pred
    loss2.append(np.mean(error**2) * 40**2)   # loss in the original y units

    # backward pass (chain rule)
    g = -2 * error                   # dLoss/dpred for each point
    dv1, dv2, dd = np.mean(g * h1), np.mean(g * h2), np.mean(g)
    g1 = g * v1 * (z1 >= 0)          # sent back through v1 and neuron 1's ReLU
    g2 = g * v2 * (z2 >= 0)
    dw1, dc1 = np.mean(g1 * xs), np.mean(g1)
    dw2, dc2 = np.mean(g2 * xs), np.mean(g2)

    # update every weight
    w1 -= lr * dw1; c1 -= lr * dc1
    w2 -= lr * dw2; c2 -= lr * dc2
    v1 -= lr * dv1; v2 -= lr * dv2; d -= lr * dd

print(f"Two neurons: loss = {loss2[-1]:.1f}")
print(f"  right arm: slope {40*v1*w1/20:+.2f}, starts at x = {60 - 20*c1/w1:.1f}")
print(f"  left arm:  slope {40*v2*w2/20:+.2f}, starts at x = {60 - 20*c2/w2:.1f}")

# ---------------- Plots ----------------
xx = np.linspace(18, 101, 300)
xxs = (xx - 60) / 20
fit1 = np.maximum(0, m * xxs + b)
fit2 = 40 * (v1 * np.maximum(0, w1 * xxs + c1) + v2 * np.maximum(0, w2 * xxs + c2) + d)

plt.figure(figsize=(7, 4.5))
plt.scatter(x, y, color="gray", s=15, label="Anna's V data")
plt.plot(xx, fit1, color="#eb6834", lw=2, label="one neuron + ReLU")
plt.plot(xx, fit2, color="#2a78d6", lw=2, label="two neurons + ReLU")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Data and fit")
plt.legend(); plt.tight_layout(); plt.savefig("v_fit.png", dpi=150)

plt.figure(figsize=(7, 4.5))
plt.plot(loss1, color="#eb6834", lw=2, label="one neuron + ReLU")
plt.plot(loss2, color="#2a78d6", lw=2, label="two neurons + ReLU")
plt.yscale("log")
plt.xlabel("epoch"); plt.ylabel("loss (log scale)"); plt.title("Loss vs epoch")
plt.legend(); plt.tight_layout(); plt.savefig("v_loss.png", dpi=150)
plt.show()
