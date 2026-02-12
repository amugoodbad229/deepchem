import torch
import torch.nn as nn
import numpy as np


class HeatConductionFEM1D(nn.Module):
    """
    1D FEM Heat Conduction Solver using proper Galerkin weak formulation.

    Solves: d/dx(k * dT/dx) + q(x) = 0 on [0,1] with T(0)=T_left, T(1)=T_right

    This implementation uses:
    - Finite Element Method (not finite difference)
    - Linear hat functions (P1 elements)
    - Gauss quadrature for integration
    - Weak form with proper assembly

    Reference: JAX-FEM (arXiv:2212.00964) and physics-informed operator learning
    """

    def __init__(self, n_elements=20):
        super().__init__()
        self.n = n_elements
        self.h = 1.0 / n_elements
        # Learnable thermal conductivity
        self.conductivity = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def assemble(self, heat_source):
        """
        Assemble stiffness matrix K and load vector F using FEM weak form.

        Weak formulation:
        ∫ k * dT/dx * dφ/dx dx = ∫ q(x) * φ(x) dx

        where φ are test functions (hat functions)
        """
        n = self.n
        h = self.h
        k = self.conductivity

        # Total nodes including boundaries
        n_nodes = n + 1
        K = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
        F = torch.zeros(n_nodes, dtype=torch.float32)

        # Element stiffness matrix for 1D linear elements
        # From weak form: ∫ k * dφ_i/dx * dφ_j/dx dx
        # For linear elements: dφ/dx = ±1/h
        ke = (k / h) * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)

        # Loop over all elements
        for elem in range(n):
            left_node = elem
            right_node = elem + 1

            # === STIFFNESS MATRIX ASSEMBLY ===
            # Add element contributions to global matrix
            K[left_node, left_node] += ke[0, 0]
            K[left_node, right_node] += ke[0, 1]
            K[right_node, left_node] += ke[1, 0]
            K[right_node, right_node] += ke[1, 1]

            # === LOAD VECTOR ASSEMBLY ===
            # F_i = ∫ q(x) * φ_i(x) dx
            # Use 2-point Gauss quadrature for accuracy
            # Gauss points: x_g = x_mid ± h/(2√3)
            x_mid = (elem + 0.5) * h
            gauss_offset = h / (2.0 * np.sqrt(3.0))

            # Two Gauss points per element
            x_gauss1 = x_mid - gauss_offset
            x_gauss2 = x_mid + gauss_offset

            q1 = heat_source(torch.tensor(x_gauss1))
            q2 = heat_source(torch.tensor(x_gauss2))

            # Shape functions at Gauss points
            # For left node: φ_left(x) = (x_{i+1} - x) / h
            # For right node: φ_right(x) = (x - x_i) / h

            # At first Gauss point
            xi1 = (x_gauss1 - elem * h) / h  # Local coordinate [0,1]
            phi_left1 = 1.0 - xi1
            phi_right1 = xi1

            # At second Gauss point
            xi2 = (x_gauss2 - elem * h) / h
            phi_left2 = 1.0 - xi2
            phi_right2 = xi2

            # Gauss quadrature weights (both 0.5 for 2-point rule on [0,1])
            weight = 0.5

            # Accumulate contributions (weight * h * q * φ)
            F[left_node] += weight * h * (q1 * phi_left1 + q2 * phi_left2)
            F[right_node] += weight * h * (q1 * phi_right1 + q2 * phi_right2)

        return K, F

    def solve_steady_state(self, T_left, T_right, heat_source):
        """
        Solve steady-state heat equation with Dirichlet boundary conditions.

        Args:
            T_left: Temperature at x=0
            T_right: Temperature at x=1
            heat_source: Function q(x) representing heat generation

        Returns:
            T: Temperature field at all nodes
        """
        K, F = self.assemble(heat_source)

        # Apply Dirichlet boundary conditions using elimination method
        # Set T(0) = T_left, T(n) = T_right

        # Modify system for boundary conditions
        # Extract interior system (nodes 1 to n-1)
        K_interior = K[1:-1, 1:-1]
        F_interior = F[1:-1].clone()

        # Account for known boundary values in RHS
        F_interior -= K[1:-1, 0] * T_left
        F_interior -= K[1:-1, -1] * T_right

        # Solve interior system
        T_interior = torch.linalg.solve(K_interior, F_interior)

        # Construct full solution
        T = torch.zeros(self.n + 1, dtype=torch.float32)
        T[0] = T_left
        T[1:-1] = T_interior
        T[-1] = T_right

        return T

    def get_x_coordinates(self):
        """Return spatial coordinates of nodes"""
        return torch.linspace(0, 1, self.n + 1)


if __name__ == "__main__":
    print("=== FEM Heat Conduction: Forward Problem ===\n")

    # Test with known conductivity
    model = HeatConductionFEM1D(n_elements=50)
    model.conductivity.data = torch.tensor(2.5)

    # Heat source
    def q(x):
        return torch.sin(np.pi * x)

    # Solve
    T = model.solve_steady_state(T_left=0.0, T_right=0.0, heat_source=q)

    print(f"Conductivity k: {model.conductivity.item():.4f}")
    print(f"Max temperature: {T.max().item():.6f}")
    print(f"Min temperature: {T.min().item():.6f}")
    print(f"Temperature at midpoint: {T[25]:.6f}\n")

    # === Inverse Problem ===
    print("=== FEM Heat Conduction: Inverse Problem ===\n")

    true_k = 3.5

    # Generate synthetic measurements (no gradient tracking)
    with torch.no_grad():
        experiment = HeatConductionFEM1D(n_elements=50)
        experiment.conductivity.data = torch.tensor(true_k)
        T_true = experiment.solve_steady_state(0.0, 0.0, q)
        # Add 1% noise proportional to signal
        noise_level = 0.01 * torch.std(T_true)
        T_measured = T_true + noise_level * torch.randn_like(T_true)

    print(f"True k: {true_k}")
    print("Initial guess: 1.0")
    print("Starting optimization...\n")

    # Create model for recovery
    model = HeatConductionFEM1D(n_elements=50)
    model.conductivity.data = torch.tensor(1.0)
    optimizer = torch.optim.Adam([model.conductivity], lr=0.5)

    for epoch in range(100):
        optimizer.zero_grad()
        T_pred = model.solve_steady_state(0.0, 0.0, q)
        loss = torch.mean((T_pred - T_measured) ** 2)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: k={model.conductivity.item():.4f}, loss={loss.item():.6f}")

    print(f"\nRecovered k: {model.conductivity.item():.4f}")
    print(f"True k: {true_k}")
    print(f"Error: {abs(model.conductivity.item() - true_k):.4f}")
    print(f"Relative error: {abs(model.conductivity.item() - true_k) / true_k * 100:.2f}%")
