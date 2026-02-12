import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


class SimpleFEM1D(nn.Module):
    """
    1D FEM Poisson solver: -k*u''(x) = f(x) on [0,1] with u(0)=u(1)=0

    Uses linear hat functions and midpoint quadrature for load vector assembly.
    """

    def __init__(self, n_elements, k=1.0):
        super().__init__()
        self.n = n_elements
        self.h = 1.0 / n_elements
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))

    def assemble(self, f_source):
        """Assemble stiffness matrix K and load vector F using midpoint rule."""
        n = self.n
        h = self.h
        k = self.k

        n_interior = n - 1
        K = torch.zeros((n_interior, n_interior), dtype=torch.float32)
        F = torch.zeros(n_interior, dtype=torch.float32)

        ke = (k / h) * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)

        for elem in range(n):
            left_node = elem
            right_node = elem + 1

            # Stiffness matrix assembly
            if left_node > 0:
                i_local = left_node - 1
                K[i_local, i_local] += ke[0, 0]

                if right_node < n:
                    j_local = right_node - 1
                    K[i_local, j_local] += ke[0, 1]
                    K[j_local, i_local] += ke[1, 0]
                    K[j_local, j_local] += ke[1, 1]
            elif right_node < n:
                j_local = right_node - 1
                K[j_local, j_local] += ke[1, 1]

            # Load vector assembly using MIDPOINT rule
            x_mid = (elem + 0.5) * h
            f_mid = f_source(torch.tensor(x_mid))

            if left_node > 0:
                i_local = left_node - 1
                F[i_local] += 0.5 * h * f_mid

            if right_node < n:
                j_local = right_node - 1
                F[j_local] += 0.5 * h * f_mid

        return K, F

    def solve(self, f_source):
        """Solve the FEM system Ku = F"""
        K, F = self.assemble(f_source)
        u_interior = torch.linalg.solve(K, F)

        u_full = torch.zeros(self.n + 1, dtype=torch.float32)
        u_full[1:-1] = u_interior

        return u_full


def analytical_solution(x, k=1.0):
    """Analytical solution to -k*u''(x) = sin(πx)"""
    return torch.sin(np.pi * x) / (np.pi**2 * k)


# Convergence study
mesh_sizes = [10, 20, 40, 80, 160]
errors_L2 = []
hs = []


def f(x):
    return torch.sin(np.pi * x)


print("Convergence Study:")
print(f"{'n':<10} {'h':<12} {'L2 Error':<15}")
print("-" * 40)

for n in mesh_sizes:
    with torch.no_grad():
        fem = SimpleFEM1D(n_elements=n, k=1.0)
        u_num = fem.solve(f)

        # Interior nodes only
        x_interior = torch.linspace(0, 1, n + 1)[1:-1]
        u_exact = analytical_solution(x_interior, k=1.0)

        h = 1.0 / n

        # L2 error on interior nodes
        error_L2 = torch.sqrt(torch.mean((u_num[1:-1] - u_exact) ** 2))

        errors_L2.append(error_L2.item())
        hs.append(h)
        print(f"{n:<10} {h:<12.6f} {error_L2.item():<15.8f}")

# Compute convergence rates
rates_L2 = []
for i in range(1, len(errors_L2)):
    rate = np.log(errors_L2[i - 1] / errors_L2[i]) / np.log(hs[i - 1] / hs[i])
    rates_L2.append(rate)

avg_rate = np.mean(rates_L2)
print(f"\nAverage Convergence Rate: {avg_rate:.4f}")
print("Expected Rate: 2.0")
print(f"Test {'PASSED ✓' if avg_rate >= 1.8 else 'FAILED ✗'}")

# Visualization
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig)

# Plot 1: Solution comparison
ax1 = fig.add_subplot(gs[0, :])
fem = SimpleFEM1D(n_elements=40, k=1.0)
u = fem.solve(f)
x_fine = torch.linspace(0, 1, 200)
x_coarse = torch.linspace(0, 1, 41)
u_exact_fine = analytical_solution(x_fine, k=1.0)

ax1.plot(x_fine.numpy(), u_exact_fine.numpy(), "k-", linewidth=2, label="Analytical")
ax1.plot(
    x_coarse.numpy(),
    u.detach().numpy(),
    "ro",
    markersize=8,
    label="FEM",
    markerfacecolor="none",
    markeredgewidth=2,
)
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("u(x)", fontsize=12)
ax1.set_title("1D Poisson Equation: -u''(x) = sin(πx)", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Convergence plot (log-log)
ax2 = fig.add_subplot(gs[1, 0])
hs_array = np.array(hs)
errors_L2_array = np.array(errors_L2)

ax2.loglog(
    hs_array, errors_L2_array, "bo-", linewidth=2, markersize=8, label="Computed Error"
)
ax2.loglog(
    hs_array,
    errors_L2_array[0] * (hs_array / hs_array[0]) ** 2,
    "k--",
    linewidth=2,
    alpha=0.7,
    label="O(h²) Reference",
)
ax2.set_xlabel("Mesh size h", fontsize=12)
ax2.set_ylabel("L2 Error", fontsize=12)
ax2.set_title(f"Convergence Rate: {avg_rate:.3f}", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which="both")

# Plot 3: Pointwise error
ax3 = fig.add_subplot(gs[1, 1])
x_interior = x_coarse[1:-1]
u_exact_coarse = analytical_solution(x_interior, k=1.0)
error_pointwise = torch.abs(u[1:-1] - u_exact_coarse)

ax3.plot(
    x_interior.numpy(),
    error_pointwise.detach().numpy(),
    "g-s",
    linewidth=2,
    markersize=6,
    label="Pointwise Error",
)
ax3.set_xlabel("x", fontsize=12)
ax3.set_ylabel("|u_exact - u_FEM|", fontsize=12)
ax3.set_title("Pointwise Error (n=40)", fontsize=13, fontweight="bold")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()

os.makedirs("Images", exist_ok=True)
plt.savefig("Images/validation_1d.png", dpi=300, bbox_inches="tight")
print("\nFigure saved: Images/validation_1d.png")
plt.show()
