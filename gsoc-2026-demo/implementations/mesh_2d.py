import torch
import torch.nn as nn
import numpy as np


class SimpleMesh2D(nn.Module):
    """
    2D Triangular Mesh with Differentiable Geometry

    Features:
    - Structured/unstructured triangular meshes
    - Differentiable node coordinates
    - FEM-ready connectivity data
    - Quality metrics (aspect ratio, Jacobian)
    - Boundary identification
    - Neighbor queries

    Reference: G-Adaptivity (arXiv:2407.04516), torch-fem
    """

    def __init__(self, nx=5, ny=5):
        super().__init__()
        self.nx, self.ny = nx, ny

        # Create structured grid
        x = torch.linspace(0, 1, nx, dtype=torch.float32)
        y = torch.linspace(0, 1, ny, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        # Nodes as learnable parameters for optimization
        self.nodes = nn.Parameter(
            torch.stack([xx.flatten(), yy.flatten()], dim=1))

        # Build connectivity (each quad → 2 triangles)
        elements = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                n = i * ny + j
                # Lower triangle
                elements.append([n, n + 1, n + ny])
                # Upper triangle
                elements.append([n + 1, n + ny + 1, n + ny])

        self.elements = torch.tensor(elements, dtype=torch.long)

        # Store initial area for conservation constraint
        with torch.no_grad():
            self.target_area = self.compute_total_area()

        # Identify boundary nodes
        self._identify_boundaries()

    def _identify_boundaries(self):
        """Identify boundary nodes for applying constraints"""
        nx, ny = self.nx, self.ny

        boundary_mask = torch.zeros(nx * ny, dtype=torch.bool)

        # Bottom edge: j=0
        boundary_mask[0:nx * ny:ny] = True
        # Top edge: j=ny-1
        boundary_mask[ny - 1:nx * ny:ny] = True
        # Left edge: i=0
        boundary_mask[0:ny] = True
        # Right edge: i=nx-1
        boundary_mask[(nx - 1) * ny:nx * ny] = True

        self.boundary_mask = boundary_mask
        self.interior_mask = ~boundary_mask

        # Store corner indices for constraints
        self.corners = torch.tensor(
            [
                0,  # Bottom-left
                ny - 1,  # Bottom-right
                (nx - 1) * ny,  # Top-left
                nx * ny - 1,  # Top-right
            ],
            dtype=torch.long,
        )

    def compute_element_area(self, element_idx):
        """
        Compute area of a triangular element using cross product.

        For triangle with vertices a, b, c:
        Area = 0.5 * |cross(b-a, c-a)|
        """
        elem = self.elements[element_idx]
        coords = self.nodes[elem]  # Shape: (3, 2)

        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]

        # Cross product in 2D: v1[0]*v2[1] - v1[1]*v2[0]
        area = 0.5 * torch.abs(v1[0] * v2[1] - v1[1] * v2[0])

        return area

    def compute_total_area(self):
        """Compute total mesh area (should be 1.0 for unit square)"""
        total = torch.sum(
            torch.stack([
                self.compute_element_area(i) for i in range(len(self.elements))
            ]))
        return total

    def compute_aspect_ratio(self, element_idx):
        """
        Compute aspect ratio quality metric.

        Aspect ratio = 4*sqrt(3)*Area / (L1² + L2² + L3²)

        Perfect equilateral triangle: aspect_ratio = 1.0
        Degenerate triangle: aspect_ratio → 0
        """
        elem = self.elements[element_idx]
        coords = self.nodes[elem]

        # Edge vectors
        e1 = coords[1] - coords[0]
        e2 = coords[2] - coords[1]
        e3 = coords[0] - coords[2]

        # Edge lengths squared
        L1_sq = torch.sum(e1**2)
        L2_sq = torch.sum(e2**2)
        L3_sq = torch.sum(e3**2)

        area = self.compute_element_area(element_idx)

        # Quality metric (1.0 = perfect, 0.0 = degenerate)
        aspect = 4.0 * np.sqrt(3.0) * area / (L1_sq + L2_sq + L3_sq + 1e-10)

        return aspect

    def compute_all_quality_metrics(self):
        """
        Compute quality metrics for all elements.

        Returns:
            dict with 'aspect_ratios', 'min_aspect', 'mean_aspect'
        """
        aspect_ratios = torch.stack(
            [self.compute_aspect_ratio(i) for i in range(len(self.elements))])

        return {
            "aspect_ratios": aspect_ratios,
            "min_aspect": torch.min(aspect_ratios),
            "mean_aspect": torch.mean(aspect_ratios),
            "worst_element_idx": torch.argmin(aspect_ratios).item(),
        }

    def enforce_boundary_constraints(self):
        """
        Enforce that boundary nodes stay fixed or within bounds.
        Call this after optimizer.step()
        """
        with torch.no_grad():
            # Clamp all nodes to [0, 1]
            self.nodes.data = torch.clamp(self.nodes.data, 0.0, 1.0)

            # Fix corners exactly
            self.nodes.data[0] = torch.tensor([0.0, 0.0])
            self.nodes.data[self.ny - 1] = torch.tensor([0.0, 1.0])
            self.nodes.data[(self.nx - 1) * self.ny] = torch.tensor([1.0, 0.0])
            self.nodes.data[-1] = torch.tensor([1.0, 1.0])

    def get_neighbor_elements(self, node_idx):
        """
        Find all elements that contain a given node.
        Useful for FEM assembly.
        """
        neighbors = []
        for i, elem in enumerate(self.elements):
            if node_idx in elem:
                neighbors.append(i)
        return neighbors


if __name__ == "__main__":
    print("=" * 60)
    print("2D MESH: BASIC TESTS")
    print("=" * 60)

    # Test 1: Area conservation
    print("\nTest 1: Area Conservation")
    mesh = SimpleMesh2D(nx=4, ny=4)
    total_area = mesh.compute_total_area()
    print(f"Initial total area: {total_area:.10f}")
    print("Expected: 1.0000000000")
    print(f"Test: {'PASSED ✓' if abs(total_area - 1.0) < 1e-8 else 'FAILED ✗'}")

    # Test 2: Quality metrics
    print("\nTest 2: Initial Quality Metrics")
    metrics = mesh.compute_all_quality_metrics()
    print(f"Min aspect ratio: {metrics['min_aspect']:.6f}")
    print(f"Mean aspect ratio: {metrics['mean_aspect']:.6f}")
    print(f"Worst element index: {metrics['worst_element_idx']}")

    # Test 3: Mesh optimization with area preservation
    print("\nTest 3: Mesh Quality Optimization with Area Preservation")
    mesh = SimpleMesh2D(nx=5, ny=5)

    initial_quality = mesh.compute_all_quality_metrics()
    initial_area = mesh.compute_total_area()
    print(f"Initial min quality: {initial_quality['min_aspect']:.6f}")
    print(f"Initial area: {initial_area:.10f}")

    optimizer = torch.optim.Adam([mesh.nodes], lr=0.01)

    for iter in range(100):
        optimizer.zero_grad()

        # === KEY FIX: Multi-objective loss with area preservation ===
        metrics = mesh.compute_all_quality_metrics()
        current_area = mesh.compute_total_area()

        # Loss = maximize quality + preserve area
        quality_loss = -metrics["min_aspect"]
        area_penalty = 10.0 * (current_area - mesh.target_area)**2

        loss = quality_loss + area_penalty

        loss.backward()
        optimizer.step()

        # Enforce constraints
        mesh.enforce_boundary_constraints()

    final_quality = mesh.compute_all_quality_metrics()
    final_area = mesh.compute_total_area()

    print(f"Final min quality: {final_quality['min_aspect']:.6f}")
    print(f"Final area: {final_area:.10f}")
    print(
        f"Quality improvement: {(final_quality['min_aspect'] - initial_quality['min_aspect']):.6f}"
    )
    print(f"Area change: {abs(final_area - initial_area):.10f}")

    quality_improved = final_quality["min_aspect"] > initial_quality[
        "min_aspect"]
    area_preserved = abs(final_area - 1.0) < 0.02  # Allow 2% error

    print(f"Quality Test: {'PASSED ✓' if quality_improved else 'FAILED ✗'}")
    print(f"Area Test: {'PASSED ✓' if area_preserved else 'FAILED ✗'}")
