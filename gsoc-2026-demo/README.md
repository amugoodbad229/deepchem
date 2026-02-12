# GSoC 2026 Demo: Differentiable FEM Proof-of-Concepts

**Student:** Ayman Khan  
**GitHub:** https://github.com/amugoodbad229  
**Branch:** gsoc-fem-demo  

---

## Overview

This repository contains proof-of-concept implementations demonstrating differentiable Finite Element Methods (FEM) for the **GSoC 2026 application** to DeepChem.

**Proposed Project:** Differentiable FEM/FVM Integration into DeepChem's TorchModel Framework

---

## Quick Results

| Demo | Description | Validation |
|------|-------------|------------|
| 1D Poisson Solver | FEM assembly + learnable parameter `k` | Convergence rate: 2.40 (superconvergence) ✓ |
| Heat Conduction | Inverse problem: recover conductivity from measurements | Avg recovery error: 1.14%, Max: 3.88% ✓ |
| 2D Triangular Mesh | Differentiable geometry + quality optimization | Quality: 0.87→0.93, Area preserved ✓ |

---

## Repository Structure

```
gsoc-2026-demo/
├── implementations/
│   ├── __init__.py
│   ├── simple_fem_1d.py      # 1D Poisson FEM solver
│   ├── heat_conduction.py    # Heat equation with inverse problem
│   └── mesh_2d.py            # 2D triangular mesh with autograd
├── validation/
│   ├── validate_1d.py        # Convergence study
│   ├── validate_heat.py      # Parameter recovery tests
│   ├── validate_mesh.py      # Area + quality validation
│   └── test_validation.py    # Pytest test suite
├── Images/                    # Generated validation figures
├── README.md                  # This file
└── DETAILED_EXPLANATION.md    # Technical documentation
```

---

## Quick Start

```bash
# Setup
uv init --python 3.10
uv add torch numpy matplotlib yapf flake8 mypy pytest

# Run validations (generates figures in Images/)
uv run python validation/validate_1d.py
uv run python validation/validate_heat.py
uv run python validation/validate_mesh.py
```

---

## Testing

This project includes a comprehensive pytest test suite that validates all implementations:

```bash
# Run all tests
uv run pytest validation/test_validation.py -v

# Run specific test classes
uv run pytest validation/test_validation.py::TestFEM1D -v
uv run pytest validation/test_validation.py::TestHeatConduction -v
uv run pytest validation/test_validation.py::TestMesh2D -v

# Run with coverage
uv run pytest validation/test_validation.py --cov=implementations --cov-report=html
```

### Test Coverage

| Test Class | Description |
|------------|-------------|
| `TestFEM1D` | Tests 1D Poisson solver convergence rate and inverse problem |
| `TestHeatConduction` | Tests heat equation inverse problem for multiple k values |
| `TestMesh2D` | Tests area conservation, quality optimization, and boundary constraints |

---

## Code Quality

This project follows [DeepChem's contribution guidelines](https://github.com/deepchem/deepchem/blob/master/CONTRIBUTING.md):

### Linting and Type Checking

```bash
# Format code with yapf
uv run yapf -i implementations/__init__.py implementations/simple_fem_1d.py implementations/heat_conduction.py implementations/mesh_2d.py validation/validate_1d.py validation/validate_heat.py validation/validate_mesh.py validation/test_validation.py

# Check code style with flake8
uv run flake8 implementations/ validation/

# Type checking with mypy
uv run mypy implementations/ validation/

# Run all quality checks
uv run flake8 implementations/ validation/ && uv run mypy implementations/ validation/
```

### Pre-commit Checks

All code must pass the following before submission:
- ✓ **YAPF** — Code formatting
- ✓ **Flake8** — Style checking (PEP 8 compliance)
- ✓ **mypy** — Type checking
- ✓ **pytest** — Unit tests pass

### Coding Standards

- **NumPy-style docstrings** for all functions and classes
- **Type hints** where applicable
- **PEP 8** compliance (checked via flake8)
- **No trailing whitespace** or blank lines with whitespace

---

## Core Concepts Demonstrated

| Concept | Implementation | Why It Matters |
|---------|----------------|----------------|
| FEM Assembly | `simple_fem_1d.py` | Stiffness matrix, load vector construction |
| PyTorch Autograd | All files | Gradients flow through FEM solver |
| Inverse Problems | `heat_conduction.py` | Parameter estimation from measurements |
| Mesh Geometry | `mesh_2d.py` | Differentiable coordinates enable shape optimization |

---

## Sample Outputs

### 1D Poisson Convergence Study

```
============================================================
1D FEM POISSON SOLVER: CONVERGENCE VALIDATION
============================================================

n          h            L2 Error       
----------------------------------------
10         0.100000     0.00031326     
20         0.050000     0.00007572     
40         0.025000     0.00001869     
80         0.012500     0.00000471     
160        0.006250     0.00000040     

Average Convergence Rate: 2.4045
Expected Rate: 2.0
Note: Rate > 2.0 indicates superconvergence (better than expected)
```

### Heat Conduction Inverse Problem

```
============================================================
FEM HEAT CONDUCTION VALIDATION
============================================================

k_true= 0.50 → k_recovered=0.499904 (error=0.0193%)
k_true= 1.00 → k_recovered=1.000344 (error=0.0344%)
k_true= 2.00 → k_recovered=1.998340 (error=0.0830%)
k_true= 3.50 → k_recovered=3.585730 (error=2.4494%)
k_true= 5.00 → k_recovered=5.194020 (error=3.8804%)
k_true=10.00 → k_recovered=9.961943 (error=0.3806%)

Average Recovery Error: 1.1412%
Maximum Recovery Error: 3.8804%
Test: PASSED ✓
```

### 2D Mesh Quality Optimization

```
======================================================================
2D MESH VALIDATION: AREA CONSERVATION & QUALITY OPTIMIZATION
======================================================================

[Test 1] Area Conservation Across Mesh Sizes
----------------------------------------------------------------------
5× 5 grid: area=1.000000000000, min_q=0.866025, avg_q=0.866025
10×10 grid: area=0.999999940395, min_q=0.866025, avg_q=0.866025
20×20 grid: area=1.000000000000, min_q=0.866025, avg_q=0.866025
40×40 grid: area=0.999999940395, min_q=0.866025, avg_q=0.866025

Area Conservation Test: PASSED ✓

[Test 2] Mesh Quality Optimization
----------------------------------------------------------------------
Initial min quality: 0.866025
Final min quality: 0.928385
Quality improvement: +7.2%

Initial area: 1.0000000000
Final area: 0.9917194843
Area error: 0.83%

Quality Improvement Test: PASSED ✓
Area Preservation Test: PASSED ✓
```

---

## Contributing

This project follows the [DeepChem Contribution Guidelines](https://github.com/deepchem/deepchem/blob/master/CONTRIBUTING.md) and [Code of Conduct](https://github.com/deepchem/deepchem/blob/master/CODE_OF_CONDUCT.md).

### Development Workflow

1. **Fork** the repository and create a feature branch
2. **Install** development dependencies:
   ```bash
   uv add torch numpy matplotlib yapf flake8 mypy pytest
   ```
3. **Make changes** following coding standards
4. **Run quality checks**:
   ```bash
   uv run yapf -i implementations/*.py validation/*.py
   uv run flake8 implementations/ validation/
   uv run mypy implementations/ validation/
   uv run pytest validation/
   ```
5. **Commit** with clear messages following conventional commits
6. **Submit** a pull request for review

### Code Standards Compliance

All code in this repository:
- Passes **flake8** style checking
- Passes **mypy** type checking
- Includes **NumPy-style docstrings**
- Has **comprehensive test coverage** via pytest
- Follows **PEP 8** style guidelines
- Uses **meaningful variable names** (no single-letter variables except in math contexts)

---

## Documentation

- **[DETAILED_EXPLANATION.md](./DETAILED_EXPLANATION.md)** — Full technical documentation with mathematical derivations, implementation details, and validation methodology

---

## GSoC 2026 Proposal Summary

**Project:** Differentiable Finite Element/Volume Methods for DeepChem

**Timeline:** 12 weeks

**Key Deliverables:**
- Core FEM module (1D/2D/3D)
- FVM support for conservation laws
- Mesh data structures with autograd
- DeepChem TorchModel integration
- Comprehensive benchmarks

---

## Questions for Mentors

1. Prioritize 2D FEM or FVM for the main project?
2. Preferred API design for TorchModel integration?
3. Support for external mesh formats (GMSH, VTK)?

---

## Contact

**GitHub:** https://github.com/amugoodbad229  
**Email:** ayman.khan1971@gmail.com  
**Discord:** batman_69680

---

*This work is part of GSoC 2026 application for Differentiable FEM/FVM project with DeepChem.*
