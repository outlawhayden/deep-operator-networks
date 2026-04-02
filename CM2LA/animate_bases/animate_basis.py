import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--qr_path",   type=str, default="../orth_qr_factors.npz")
parser.add_argument("--meta_path", type=str, default="../orth_meta.npz")
parser.add_argument("--basis_idx", type=int, default=0,   help="which basis function to animate")
parser.add_argument("--fps",       type=int, default=15)
parser.add_argument("--output",    type=str, default="basis_animation.gif")
args = parser.parse_args()

qr   = np.load(args.qr_path)
meta = np.load(args.meta_path)

Q_flat = qr["Q"]        # (T*X, K)
t_grid = meta["t_grid"] # (T,)
x_grid = meta["x_grid"] # (X,)

T, X, K = len(t_grid), len(x_grid), Q_flat.shape[1]
Q = Q_flat.reshape(T, X, K)  # (T, X, K)

k = args.basis_idx
Q_k = Q[:, :, k]             # (T, X)

abs_max = max(abs(Q_k.min()), abs(Q_k.max()))

fig, ax = plt.subplots(figsize=(7, 4))
line, = ax.plot(x_grid, Q_k[0], color="steelblue", lw=1.5)
ax.set_xlim(x_grid[0], x_grid[-1])
ax.set_ylim(-abs_max * 1.15, abs_max * 1.15)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("x")
ax.set_ylabel(f"Q[t, x, {k}]")
title = ax.set_title(f"Basis {k}  —  t = {t_grid[0]:.4f}")
ax.grid(True, alpha=0.3)
fig.tight_layout()

def update(frame):
    line.set_ydata(Q_k[frame])
    title.set_text(f"Basis {k}  —  t = {t_grid[frame]:.4f}")
    return line, title

ani = animation.FuncAnimation(
    fig, update, frames=T, interval=1000 // args.fps, blit=True
)

ani.save(args.output, writer="pillow", fps=args.fps)
plt.close(fig)
print(f"saved: {args.output}")