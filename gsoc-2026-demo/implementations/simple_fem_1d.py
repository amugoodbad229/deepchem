import torch
import torch.nn as nn
import numpy as np


class SimpleFEM1D(nn.Module):
    """
    1D FEM Poisson solver: -k*u''(x) = f(x) on [0,1] with u(0)=u(1)=0

    Uses linear hat functions and midpoint quadrature for load vector assembly.
    """

    def __init__(self, n_elements, k=1.0):
        super().__init__()
        self.n = n_elements
        self.h = 1.0 / n_elements
        # Make k a learnable parameter for inverse problems
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))

    def assemble(self, f_source):
        """
        Assemble stiffness matrix K and load vector F.

        For linear 1D elements:
        - K_ij = integral of k * phi_i' * phi_j' dx
        - F_i = integral of f(x) * phi_i(x) dx

        Uses midpoint rule for load vector integration.
        """
        n = self.n
        h = self.h
        k = self.k

        # We solve for interior nodes only (boundary nodes u(0)=u(1)=0)
        n_interior = n - 1
        K = torch.zeros((n_interior, n_interior), dtype=torch.float32)
        F = torch.zeros(n_interior, dtype=torch.float32)

        # Element stiffness matrix for 1D linear elements
        # ke = (k/h) * [[1, -1], [-1, 1]]
        ke = (k / h) * torch.tensor([[1.0, -1.0], [-1.0, 1.0]],
                                    dtype=torch.float32)

        # Loop over all elements [x_i, x_{i+1}]
        for elem in range(n):
            # Global node indices for this element
            left_node = elem  # x_i
            right_node = elem + 1  # x_{i+1}

            # === STIFFNESS MATRIX ASSEMBLY ===
            # Map global nodes to interior indices (subtract 1 since we exclude boundaries)
            # Only assemble contributions from nodes that are NOT on boundaries

            if left_node > 0:  # left node is interior
                i_local = left_node - 1
                K[i_local, i_local] += ke[0, 0]

                if right_node < n:  # right node is also interior
                    j_local = right_node - 1
                    K[i_local, j_local] += ke[0, 1]
                    K[j_local, i_local] += ke[1, 0]
                    K[j_local, j_local] += ke[1, 1]
            elif right_node < n:  # only right node is interior (left is boundary)
                j_local = right_node - 1
                K[j_local, j_local] += ke[1, 1]

            # === LOAD VECTOR ASSEMBLY (CORRECT METHOD) ===
            # F_i = integral of f(x) * phi_i(x) dx over element
            #
            # For linear elements, we use midpoint quadrature:
            # integral_{x_i}^{x_{i+1}} f(x)*phi(x) dx ≈ h * f(x_mid) * phi(x_mid)
            #
            # At the midpoint x_mid = x_i + h/2:
            # - phi_{left}(x_mid) = 0.5 (hat function from left node)
            # - phi_{right}(x_mid) = 0.5 (hat function from right node)

            x_mid = (elem + 0.5) * h  # Midpoint of element [x_i, x_{i+1}]
            f_mid = f_source(torch.tensor(x_mid))

            # Contribution to left node: h * f(x_mid) * phi_left(x_mid) = h * f_mid * 0.5
            if left_node > 0:  # left node is interior
                i_local = left_node - 1
                F[i_local] += 0.5 * h * f_mid

            # Contribution to right node: h * f(x_mid) * phi_right(x_mid) = h * f_mid * 0.5
            if right_node < n:  # right node is interior
                j_local = right_node - 1
                F[j_local] += 0.5 * h * f_mid

        return K, F

    def solve(self, f_source):
        """Solve the FEM system Ku = F"""
        K, F = self.assemble(f_source)

        # Solve linear system
        u_interior = torch.linalg.solve(K, F)

        # Construct full solution with boundary conditions
        u_full = torch.zeros(self.n + 1, dtype=torch.float32)
        u_full[1:-1] = u_interior  # Interior nodes
        # u_full[0] = 0, u_full[-1] = 0 (already zero, Dirichlet BC)

        return u_full


def analytical_solution(x, k=1.0):
    """
    Analytical solution to -k*u''(x) = sin(πx) with u(0)=u(1)=0
    Solution: u(x) = sin(πx) / (π²*k)
    """
    return torch.sin(np.pi * x) / (np.pi**2 * k)


if __name__ == "__main__":
    print("=== Forward Solve Test ===")

    def source_func(x):
        return torch.sin(np.pi * x)

    # Test with different mesh sizes
    for n in [10, 40]:
        fem = SimpleFEM1D(n_elements=n, k=1.0)
        u = fem.solve(source_func)

        # Compare with analytical solution
        x = torch.linspace(0, 1, n + 1)
        u_exact = analytical_solution(x, k=1.0)

        # Compute error at interior nodes
        error_interior = torch.norm(u[1:-1] - u_exact[1:-1]) / torch.norm(
            u_exact[1:-1])

        if n == 10:
            print(f"Solution at nodes (n={n}): {u[1:-1]}")
            mid_idx = n // 2
            print(f"Solution at midpoint (index {mid_idx}): {u[mid_idx]:.4f}")
            print(f"Expected analytical at x=0.5: {u_exact[mid_idx]:.4f}")

        print(f"Relative L2 error (n={n}): {error_interior:.6f}")

    # Inverse problem
    print("\n=== Inverse Problem Test ===")

    true_k = 2.0

    # Generate target (without tracking gradients)
    with torch.no_grad():
        fem_true = SimpleFEM1D(n_elements=10, k=true_k)
        target = fem_true.solve(source_func)

    print(f"Target solution: {target[1:-1]}")
    print(f"True k: {true_k}")

    # Create new model for recovery
    fem = SimpleFEM1D(n_elements=10, k=1.0)
    optimizer = torch.optim.Adam([fem.k], lr=0.1)

    print(f"Initial k: {fem.k.item():.4f}")

    for epoch in range(200):
        optimizer.zero_grad()
        u_pred = fem.solve(source_func)
        loss = torch.mean((u_pred - target)**2)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}: k={fem.k.item():.4f}, loss={loss.item():.6f}")

    print(f"\nRecovered k: {fem.k.item():.4f}")
    print(f"True k: {true_k}")
    print(f"Error: {abs(fem.k.item() - true_k):.4f}")
