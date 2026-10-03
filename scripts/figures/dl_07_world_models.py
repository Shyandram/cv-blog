"""Controlled dynamics, rollout and planning figures; synthetic, CPU-only.

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/dl_07_world_models.py
The state is observed directly. This illustrates dynamics and planning, not
learning visual representations or reproducing a published benchmark.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def experiment():
    rng = np.random.default_rng(7)
    A = np.array([[1.0, 0.1], [0.0, 0.94]])
    B = np.array([0.005, 0.1])

    def collect(n, steps=40):
        states = np.zeros((n, steps + 1, 2))
        states[:, 0] = rng.normal(size=(n, 2)) * [0.8, 0.3]
        actions = rng.uniform(-1, 1, size=(n, steps))
        for t in range(steps):
            states[:, t + 1] = (
                states[:, t] @ A.T + actions[:, t, None] * B
                + rng.normal(scale=0.002, size=(n, 2))
            )
        return states, actions

    train, train_u = collect(120)
    test, test_u = collect(80)  # independent trajectories, never used to fit
    phi = np.column_stack([train[:, :-1].reshape(-1, 2), train_u.ravel()])
    target = train[:, 1:].reshape(-1, 2)
    W = np.linalg.lstsq(phi, target, rcond=None)[0]
    baseline_W = np.linalg.lstsq(phi[:, :2], target, rcond=None)[0]

    def predict(x, u):
        return np.concatenate([x, np.asarray(u)[..., None]], axis=-1) @ W

    one_step = predict(test[:, :-1], test_u)
    free = np.zeros_like(test)
    free[:, 0] = test[:, 0]
    for t in range(test_u.shape[1]):
        free[:, t + 1] = predict(free[:, t], test_u[:, t])
    horizons = np.arange(1, test.shape[1])
    one_error = np.sqrt(np.mean((one_step - test[:, 1:]) ** 2, axis=(0, 2)))
    free_error = np.sqrt(np.mean((free[:, 1:] - test[:, 1:]) ** 2, axis=(0, 2)))
    baseline = test[:, :-1] @ baseline_W
    print("test one-step RMSE:", np.sqrt(np.mean((one_step - test[:, 1:]) ** 2)))
    print("action-free RMSE:", np.sqrt(np.mean((baseline - test[:, 1:]) ** 2)))
    print("40-step rollout RMSE:", free_error[-1])

    def plan(x, mean, horizon=20, population=256):
        std = np.ones(horizon)
        for _ in range(4):
            candidates = np.clip(rng.normal(mean, std, (population, horizon)), -1, 1)
            imagined = np.broadcast_to(x, (population, 2)).copy()
            costs = np.zeros(population)
            for t in range(horizon):
                imagined = predict(imagined, candidates[:, t])
                costs += (imagined[:, 0] ** 2 + 0.15 * imagined[:, 1] ** 2
                          + 0.02 * candidates[:, t] ** 2)
            costs += 20 * imagined[:, 0] ** 2
            elite = candidates[np.argsort(costs)[:32]]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 0.05
        return mean

    controlled, unforced, applied = [np.array([1.5, 0.0])], [np.array([1.5, 0.0])], []
    mean = np.zeros(20)
    for _ in range(40):
        sequence = plan(controlled[-1], mean)
        u = sequence[0]
        applied.append(u)
        # Execute only one action in the real synthetic system.
        controlled.append(A @ controlled[-1] + B * u)
        unforced.append(A @ unforced[-1])
        mean = np.r_[sequence[1:], 0.0]
    controlled, unforced = np.array(controlled), np.array(unforced)
    print("final position: controlled / zero action:", controlled[-1, 0], unforced[-1, 0])
    assert np.sqrt(np.mean((one_step - test[:, 1:]) ** 2)) < 0.01
    assert abs(controlled[-1, 0]) < abs(unforced[-1, 0])
    return horizons, one_error, free_error, controlled, unforced, np.array(applied)


if __name__ == "__main__":
    h, one, free, control, zero, actions = experiment()
    out = Path(__file__).resolve().parents[2] / "src/assets/deep-learning/dl-07-world-models"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(h, one, label="One-step: true state supplied every step", color="#2a6fdb")
    ax.plot(h, free, label="Open-loop: only initial state supplied", color="#e4572e")
    ax.set(xlabel="Prediction horizon", ylabel="RMSE over independent test trajectories",
           title="Small one-step error can accumulate during imagination")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.savefig(out / "rollout-error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(control[:, 0], label="Learned-model MPC")
    axes[0].plot(zero[:, 0], "--", label="Zero action")
    axes[0].axhline(0, color="gray", linewidth=1)
    axes[0].set(xlabel="Executed step", ylabel="Position", title="Approach a goal at position zero")
    axes[0].legend(fontsize=9)
    axes[1].step(np.arange(len(actions)), actions, where="post", color="#e4572e")
    axes[1].set(xlabel="Executed step", ylabel="Applied action", ylim=(-1.1, 1.1),
                title="Plan 20 steps; execute one; observe again")
    for ax in axes: ax.grid(alpha=0.2)
    fig.savefig(out / "planning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
