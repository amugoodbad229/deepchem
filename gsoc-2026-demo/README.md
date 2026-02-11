# GSOC 2026 Demo: Differentiable FEM Proof-of-Concepts

**Student:** Ayman Khan  
**GitHub:** https://github.com/amugoodbad229  
**Branch:** gsoc-fem-demo  
**Purpose:** Demonstrate understanding of differentiable FEM for GSOC application

---

## My Actual GSOC Proposal

**Project:** Differentiable Finite Element/Volume Methods for DeepChem  
**Scope:** Full FEM/FVM integration with DeepChem's TorchModel  
**Timeline:** 12 weeks  

**Key Deliverables:**
- Core FEM module (1D/2D/3D)
- FVM support for conservation laws
- Mesh data structures
- Integration with DeepChem TorchModel
- Comprehensive benchmarks

---

## These Demonstrations

The 3 implementations in this branch prove I can execute the core concepts BEFORE GSOC starts.

### Why These Demos Matter

Before implementing full FEM/FVM for DeepChem, I prove I understand:

1. **FEM Assembly** - How to build stiffness matrices and load vectors
2. **Autograd Integration** - Making FEM differentiable via PyTorch
3. **Inverse Problems** - Using gradients to optimize physical parameters
4. **Mesh Structures** - Handling geometric data with autograd support
5. **Validation Methods** - Research-grade testing and verification

### Key Results

| Demo | Lines | Validation Result |
|------|-------|-------------------|
| 1D Poisson | 48 | O(h²) convergence (rate = 2.00) ✓ |
| Heat Conduction | 45 | Parameter recovery < 0.1% error ✓ |
| 2D Mesh | 50 | Area conservation exact ✓ |

---

## How to Run

```bash
# Setup
uv init --python 3.10
uv add torch numpy matplotlib

# Run implementations
uv run python implementations/simple_fem_1d.py
uv run python implementations/heat_conduction.py
uv run python implementations/mesh_2d.py

# Run validations (generates PNG files)
uv run python validation/validate_1d.py
uv run python validation/validate_heat.py
uv run python validation/validate_mesh.py
```

---

## Questions for Mentors

1. Should I prioritize 2D FEM or FVM for the main project?
2. Preferred API design for DeepChem TorchModel integration?
3. Support for external mesh formats (GMSH)?

**Contact:** [Your email/Discord]
