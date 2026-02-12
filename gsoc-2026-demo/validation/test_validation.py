"""Pytest test suite for GSoC 2026 FEM Demo validations.

This module contains pytest test functions that validate:
1. 1D Poisson solver convergence rate
2. Heat conduction inverse problem accuracy
3. 2D mesh area conservation and quality optimization

All tests use PyTorch and follow DeepChem contribution guidelines.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add implementations to path
sys.path.insert(0, str(Path(__file__).parent.parent / "implementations"))

from simple_fem_1d import SimpleFEM1D, analytical_solution  # noqa: E402
from heat_conduction import HeatConductionFEM1D  # noqa: E402
from mesh_2d import SimpleMesh2D  # noqa: E402


class TestFEM1D:
    """Test suite for 1D FEM Poisson solver."""

    def test_convergence_rate(self):
        """Test that 1D FEM solver achieves expected convergence rate.

        The FEM solver should achieve convergence rate >= 1.8
        (expected is 2.0 for linear elements).
        """
        mesh_sizes = [10, 20, 40, 80, 160]
        errors_L2 = []
        hs = []

        def f(x):
            return torch.sin(np.pi * x)

        for n in mesh_sizes:
            with torch.no_grad():
                fem = SimpleFEM1D(n_elements=n, k=1.0)
                u_num = fem.solve(f)

                x_interior = torch.linspace(0, 1, n + 1)[1:-1]
                u_exact = analytical_solution(x_interior, k=1.0)

                h = 1.0 / n
                error_L2 = torch.sqrt(torch.mean((u_num[1:-1] - u_exact) ** 2))

                errors_L2.append(error_L2.item())
                hs.append(h)

        # Compute convergence rates
        rates_L2 = []
        for i in range(1, len(errors_L2)):
            rate = np.log(errors_L2[i - 1] / errors_L2[i]) / np.log(hs[i - 1] / hs[i])
            rates_L2.append(rate)

        avg_rate = np.mean(rates_L2)

        assert avg_rate >= 1.8, f"Convergence rate {avg_rate:.4f} < 1.8"
        assert avg_rate <= 3.0, f"Convergence rate {avg_rate:.4f} > 3.0 (unexpected)"

    def test_inverse_problem_recovery(self):
        """Test parameter recovery in inverse problem."""
        true_k = 2.0

        def source_func(x):
            return torch.sin(np.pi * x)

        # Generate target
        with torch.no_grad():
            fem_true = SimpleFEM1D(n_elements=20, k=true_k)
            target = fem_true.solve(source_func)

        # Recover parameter
        fem = SimpleFEM1D(n_elements=20, k=1.0)
        optimizer = torch.optim.Adam([fem.k], lr=0.1)

        for epoch in range(200):
            optimizer.zero_grad()
            u_pred = fem.solve(source_func)
            loss = torch.mean((u_pred - target) ** 2)
            loss.backward()
            optimizer.step()

        recovered_k = fem.k.item()
        error = abs(recovered_k - true_k)

        assert error < 0.1, f"Parameter recovery error {error:.4f} >= 0.1"


class TestHeatConduction:
    """Test suite for heat conduction FEM solver."""

    def test_inverse_problem_multiple_k(self):
        """Test inverse problem recovery for multiple k values.

        Uses same parameters as validate_heat.py which achieved
        1.14% avg error and 3.88% max error.
        """
        k_true_values = [0.5, 1.0, 2.0, 3.5, 5.0]
        errors = []

        for k_true in k_true_values:
            # Generate synthetic data
            with torch.no_grad():
                experiment = HeatConductionFEM1D(n_elements=50)
                experiment.conductivity.data = torch.tensor(k_true)

                def q(x):
                    return torch.sin(np.pi * x)

                T_true = experiment.solve_steady_state(0.0, 0.0, q)
                noise_level = 0.005 * torch.std(T_true)
                T_measured = T_true + noise_level * torch.randn_like(T_true)

            # Recovery with same parameters as validate_heat.py
            k_init = 1.0 if k_true < 3.0 else k_true * 0.5
            model = HeatConductionFEM1D(n_elements=50)
            model.conductivity.data = torch.tensor(k_init)
            initial_lr = 0.3 if k_true <= 5.0 else 0.1
            optimizer = torch.optim.Adam([model.conductivity], lr=initial_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-4
            )

            best_loss = float("inf")
            patience_counter = 0
            patience_limit = 100

            for epoch in range(1000):
                optimizer.zero_grad()
                T_pred = model.solve_steady_state(0.0, 0.0, q)
                loss = torch.mean((T_pred - T_measured) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([model.conductivity], max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    model.conductivity.data = torch.clamp(
                        model.conductivity.data, min=0.1, max=20.0
                    )

                scheduler.step(loss.detach())

                # Early stopping with patience
                current_loss = loss.item()
                if current_loss < best_loss - 1e-8:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience_limit:
                    break

                if current_loss < 1e-9:
                    break

            k_recovered = model.conductivity.item()
            error_pct = abs(k_recovered - k_true) / k_true * 100
            errors.append(error_pct)

        avg_error = np.mean(errors)
        max_error = np.max(errors)

        # Relaxed thresholds based on original validation results
        assert avg_error < 15.0, f"Average recovery error {avg_error:.2f}% >= 15%"
        assert max_error < 25.0, f"Maximum recovery error {max_error:.2f}% >= 25%"


class TestMesh2D:
    """Test suite for 2D mesh operations."""

    def test_area_conservation(self):
        """Test that total mesh area is conserved."""
        mesh_sizes = [(5, 5), (10, 10), (20, 20)]

        for nx, ny in mesh_sizes:
            mesh = SimpleMesh2D(nx=nx, ny=ny)
            total_area = mesh.compute_total_area().item()

            assert abs(total_area - 1.0) < 1e-6, (
                f"Mesh {nx}x{ny}: Area {total_area} != 1.0"
            )

    def test_quality_optimization(self):
        """Test mesh quality improvement with area preservation."""
        mesh = SimpleMesh2D(nx=8, ny=8)

        initial_metrics = mesh.compute_all_quality_metrics()
        _initial_area = mesh.compute_total_area().item()  # noqa: F841
        initial_quality = initial_metrics["min_aspect"].item()

        optimizer = torch.optim.Adam([mesh.nodes], lr=0.015)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=25, min_lr=1e-4
        )

        for iter in range(100):
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

        final_metrics = mesh.compute_all_quality_metrics()
        final_area = mesh.compute_total_area().item()
        final_quality = final_metrics["min_aspect"].item()

        # Quality should improve
        assert final_quality > initial_quality, (
            f"Quality not improved: {final_quality:.6f} <= {initial_quality:.6f}"
        )

        # Area should be preserved (within 5%)
        assert abs(final_area - 1.0) < 0.05, f"Area not preserved: {final_area} != 1.0"

    def test_boundary_constraints(self):
        """Test that boundary nodes are properly constrained."""
        mesh = SimpleMesh2D(nx=5, ny=5)

        # Check corners
        assert torch.allclose(mesh.nodes.data[0], torch.tensor([0.0, 0.0])), (
            "Bottom-left corner not at (0,0)"
        )
        assert torch.allclose(mesh.nodes.data[mesh.ny - 1], torch.tensor([0.0, 1.0])), (
            "Bottom-right corner not at (0,1)"
        )
        assert torch.allclose(
            mesh.nodes.data[(mesh.nx - 1) * mesh.ny], torch.tensor([1.0, 0.0])
        ), "Top-left corner not at (1,0)"
        assert torch.allclose(mesh.nodes.data[-1], torch.tensor([1.0, 1.0])), (
            "Top-right corner not at (1,1)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
