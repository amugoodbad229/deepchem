import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import os


class SimpleMesh2D(nn.Module):
    """2D Triangular Mesh with Differentiable Geometry"""

    def __init__(self, nx=5, ny=5):
        super().__init__()
        self.nx, self.ny = nx, ny

        x = torch.linspace(0, 1, nx, dtype=torch.float32)
        y = torch.linspace(0, 1, ny, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        self.nodes = nn.Parameter(torch.stack([xx.flatten(), yy.flatten()], dim=1))

        elements = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                n = i * ny + j
                elements.append([n, n + 1, n + ny])
                elements.append([n + 1, n + ny + 1, n + ny])

        self.elements = torch.tensor(elements, dtype=torch.long)

        # Store target area
        with torch.no_grad():
            self.target_area = self.compute_total_area()

        self._identify_boundaries()

    def _identify_boundaries(self):
        """Identify boundary nodes"""
        nx, ny = self.nx, self.ny
        boundary_mask = torch.zeros(nx * ny, dtype=torch.bool)
        boundary_mask[0:nx * ny:ny] = True
        boundary_mask[ny - 1:nx * ny:ny] = True
        boundary_mask[0:ny] = True
        boundary_mask[(nx - 1) * ny:nx * ny] = True

        self.boundary_mask = boundary_mask
        self.corners = torch.tensor(
            [0, ny - 1, (nx - 1) * ny, nx * ny - 1], dtype=torch.long
        )

    def compute_element_area(self, element_idx):
        """Compute triangle area"""
        elem = self.elements[element_idx]
        coords = self.nodes[elem]
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        area = 0.5 * torch.abs(v1[0] * v2[1] - v1[1] * v2[0])
        return area

    def compute_total_area(self):
        """Total mesh area"""
        return torch.sum(
            torch.stack(
                [self.compute_element_area(i) for i in range(len(self.elements))]
            )
        )

    def compute_aspect_ratio(self, element_idx):
        """Aspect ratio quality metric"""
        elem = self.elements[element_idx]
        coords = self.nodes[elem]

        e1 = coords[1] - coords[0]
        e2 = coords[2] - coords[1]
        e3 = coords[0] - coords[2]

        L1_sq = torch.sum(e1**2)
        L2_sq = torch.sum(e2**2)
        L3_sq = torch.sum(e3**2)

        area = self.compute_element_area(element_idx)
        aspect = 4.0 * np.sqrt(3.0) * area / (L1_sq + L2_sq + L3_sq + 1e-10)

        return aspect

    def compute_all_quality_metrics(self):
        """Compute all quality metrics"""
        aspect_ratios = torch.stack(
            [self.compute_aspect_ratio(i) for i in range(len(self.elements))]
        )

        return {
            "aspect_ratios": aspect_ratios,
            "min_aspect": torch.min(aspect_ratios),
            "mean_aspect": torch.mean(aspect_ratios),
        }

    def enforce_boundary_constraints(self):
        """Fix boundary nodes"""
        with torch.no_grad():
            self.nodes.data = torch.clamp(self.nodes.data, 0.0, 1.0)
            self.nodes.data[0] = torch.tensor([0.0, 0.0])
            self.nodes.data[self.ny - 1] = torch.tensor([0.0, 1.0])
            self.nodes.data[(self.nx - 1) * self.ny] = torch.tensor([1.0, 0.0])
            self.nodes.data[-1] = torch.tensor([1.0, 1.0])


def safe_histogram(ax, data, color, edgecolor, alpha, label):
    """Plot histogram or bar chart depending on data spread."""
    data_range = np.ptp(data)
    if data_range < 1e-4:
        # Data is effectively uniform — bar chart instead
        ax.bar(
            [np.mean(data)],
            [len(data)],
            width=0.05,
            alpha=alpha,
            color=color,
            edgecolor=edgecolor,
            label=label,
        )
        ax.set_xlim([np.mean(data) - 0.1, np.mean(data) + 0.1])
    else:
        # Compute sensible bin count from rounded unique values
        n_unique = len(np.unique(np.round(data, 4)))
        n_bins = min(20, max(5, n_unique))
        ax.hist(
            data,
            bins=n_bins,
            alpha=alpha,
            color=color,
            edgecolor=edgecolor,
            label=label,
        )


print("=" * 70)
print("2D MESH VALIDATION: AREA CONSERVATION & QUALITY OPTIMIZATION")
print("=" * 70)

# Test 1: Area conservation across mesh sizes
print("\n[Test 1] Area Conservation Across Mesh Sizes")
print("-" * 70)

mesh_sizes = [(5, 5), (10, 10), (20, 20), (40, 40)]
area_results = []

for nx, ny in mesh_sizes:
    mesh = SimpleMesh2D(nx=nx, ny=ny)
    total_area = mesh.compute_total_area().item()
    metrics = mesh.compute_all_quality_metrics()

    area_results.append(
        {
            "nx": nx,
            "ny": ny,
            "area": total_area,
            "min_aspect": metrics["min_aspect"].item(),
            "mean_aspect": metrics["mean_aspect"].item(),
        }
    )

    print(
        f"{nx}×{ny:2d} grid: area={total_area:.12f}, "
        f"min_q={metrics['min_aspect'].item():.6f}, "
        f"avg_q={metrics['mean_aspect'].item():.6f}"
    )

all_areas_correct = all(abs(r["area"] - 1.0) < 1e-6 for r in area_results)
print(f"\nArea Conservation Test: {'PASSED ✓' if all_areas_correct else 'FAILED ✗'}")

# Test 2: Mesh quality optimization
print("\n[Test 2] Mesh Quality Optimization with Area Preservation")
print("-" * 70)

mesh = SimpleMesh2D(nx=8, ny=8)

initial_metrics = mesh.compute_all_quality_metrics()
initial_area = mesh.compute_total_area().item()
print(f"Initial min quality: {initial_metrics['min_aspect'].item():.6f}")
print(f"Initial mean quality: {initial_metrics['mean_aspect'].item():.6f}")
print(f"Initial area: {initial_area:.10f}")

# Optimization with area preservation
optimizer = torch.optim.Adam([mesh.nodes], lr=0.015)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=25, min_lr=1e-4
)

quality_history = []
mean_quality_history = []
area_history = []

print("\nOptimizing mesh quality...")
for iter in range(200):
    optimizer.zero_grad()

    metrics = mesh.compute_all_quality_metrics()
    current_area = mesh.compute_total_area()

    quality_loss = -metrics["min_aspect"]
    area_penalty = 20.0 * (current_area - mesh.target_area) ** 2

    loss = quality_loss + area_penalty

    loss.backward()
    torch.nn.utils.clip_grad_norm_([mesh.nodes], max_norm=0.1)
    optimizer.step()

    mesh.enforce_boundary_constraints()

    scheduler.step(metrics["min_aspect"].detach())

    quality_history.append(metrics["min_aspect"].item())
    mean_quality_history.append(metrics["mean_aspect"].item())
    area_history.append(current_area.item())

    if iter % 50 == 0:
        print(
            f"  Iter {iter:3d}: min_q={metrics['min_aspect'].item():.6f}, "
            f"mean_q={metrics['mean_aspect'].item():.6f}, "
            f"area={current_area.item():.8f}"
        )

final_metrics = mesh.compute_all_quality_metrics()
final_area = mesh.compute_total_area().item()

print(f"\nFinal min quality: {final_metrics['min_aspect'].item():.6f}")
print(f"Final mean quality: {final_metrics['mean_aspect'].item():.6f}")
print(f"Final area: {final_area:.10f}")
print(
    f"Quality improvement: {(final_metrics['min_aspect'] - initial_metrics['min_aspect']).item():.6f}"
)
print(f"Area error: {abs(final_area - 1.0):.10f}")

quality_improved = final_metrics["min_aspect"] > initial_metrics["min_aspect"]
area_preserved = abs(final_area - 1.0) < 0.02

print(f"\nQuality Improvement Test: {'PASSED ✓' if quality_improved else 'FAILED ✗'}")
print(f"Area Preservation Test: {'PASSED ✓' if area_preserved else 'FAILED ✗'}")

# Visualization
print("\n[Generating Visualization]")
print("-" * 70)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# Initial mesh
ax1 = fig.add_subplot(gs[0, 0])
mesh_init = SimpleMesh2D(nx=8, ny=8)
nodes_init = mesh_init.nodes.detach().numpy()
elements_init = mesh_init.elements.numpy()

for elem in elements_init:
    coords = nodes_init[elem]
    poly = Polygon(
        coords,
        closed=True,
        edgecolor="black",
        linewidth=0.5,
        facecolor="lightblue",
        alpha=0.6,
    )
    ax1.add_patch(poly)

ax1.plot(nodes_init[:, 0], nodes_init[:, 1], "ko", markersize=3)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_aspect("equal")
ax1.set_title("Initial Mesh (8×8)", fontsize=14, fontweight="bold")
ax1.set_xlabel("x", fontsize=11)
ax1.set_ylabel("y", fontsize=11)
ax1.grid(True, alpha=0.2, linestyle="--")

# Optimized mesh with quality coloring
ax2 = fig.add_subplot(gs[0, 1])
nodes_opt = mesh.nodes.detach().numpy()
elements_opt = mesh.elements.numpy()
aspect_ratios = final_metrics["aspect_ratios"].detach().numpy()

for i, elem in enumerate(elements_opt):
    coords = nodes_opt[elem]
    color = cm.RdYlGn(aspect_ratios[i])  # type: ignore
    poly = Polygon(
        coords,
        closed=True,
        edgecolor="black",
        linewidth=0.5,
        facecolor=color,
        alpha=0.8,
    )
    ax2.add_patch(poly)

ax2.plot(nodes_opt[:, 0], nodes_opt[:, 1], "ko", markersize=3)
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.set_aspect("equal")
ax2.set_title("Optimized Mesh (Quality-Colored)", fontsize=14, fontweight="bold")
ax2.set_xlabel("x", fontsize=11)
ax2.set_ylabel("y", fontsize=11)
ax2.grid(True, alpha=0.2, linestyle="--")

# Add colorbar
sm = cm.ScalarMappable(
    cmap=cm.RdYlGn,  # type: ignore
    norm=plt.Normalize(vmin=0, vmax=1),
)  # type: ignore
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Aspect Ratio Quality", fontsize=10)

# Quality and area convergence
ax3 = fig.add_subplot(gs[1, :])
iterations = range(len(quality_history))
ax3_twin = ax3.twinx()

line1 = ax3.plot(iterations, quality_history, "b-", linewidth=2, label="Min Quality")
line2 = ax3.plot(
    iterations, mean_quality_history, "g--", linewidth=2, label="Mean Quality"
)
line3 = ax3_twin.plot(iterations, area_history, "r:", linewidth=2, label="Total Area")

ax3.axhline(
    y=initial_metrics["min_aspect"].item(),
    color="blue",
    linestyle=":",
    linewidth=1.5,
    alpha=0.5,
)
ax3_twin.axhline(
    y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="Target Area"
)

ax3.set_xlabel("Iteration", fontsize=12, fontweight="bold")
ax3.set_ylabel(
    "Element Quality (Aspect Ratio)", fontsize=12, fontweight="bold", color="b"
)
ax3_twin.set_ylabel("Total Area", fontsize=12, fontweight="bold", color="r")
ax3.set_title(
    "Mesh Optimization: Quality & Area Convergence", fontsize=14, fontweight="bold"
)

lines = line1 + line2 + line3
labels = [str(ln.get_label()) for ln in lines]
ax3.legend(lines, labels, fontsize=10, loc="lower right")

ax3.tick_params(axis="y", labelcolor="b")
ax3_twin.tick_params(axis="y", labelcolor="r")
ax3.grid(True, alpha=0.3, linestyle="--")

# Quality histogram comparison — initial
ax4 = fig.add_subplot(gs[2, 0])
init_aspects = mesh_init.compute_all_quality_metrics()["aspect_ratios"].detach().numpy()

safe_histogram(
    ax4,
    init_aspects,
    color="lightcoral",
    edgecolor="black",
    alpha=0.7,
    label="Initial",
)
ax4.axvline(
    x=np.mean(init_aspects), color="red", linestyle="--", linewidth=2, label="Mean"
)
ax4.set_xlabel("Aspect Ratio Quality", fontsize=11, fontweight="bold")
ax4.set_ylabel("Count", fontsize=11, fontweight="bold")
ax4.set_title("Initial Quality Distribution", fontsize=13, fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis="y")

# Quality histogram comparison — optimized
ax5 = fig.add_subplot(gs[2, 1])

safe_histogram(
    ax5,
    aspect_ratios,
    color="lightgreen",
    edgecolor="black",
    alpha=0.7,
    label="Optimized",
)
ax5.axvline(
    x=np.mean(aspect_ratios), color="green", linestyle="--", linewidth=2, label="Mean"
)
ax5.set_xlabel("Aspect Ratio Quality", fontsize=11, fontweight="bold")
ax5.set_ylabel("Count", fontsize=11, fontweight="bold")
ax5.set_title("Optimized Quality Distribution", fontsize=13, fontweight="bold")
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "2D Mesh: Differentiable Geometry & Quality Optimization",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

os.makedirs("Images", exist_ok=True)
plt.savefig("Images/validation_mesh.png", dpi=300, bbox_inches="tight")
print("Figure saved: Images/validation_mesh.png")
print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
plt.show()
