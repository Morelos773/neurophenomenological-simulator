# Neurophenomenological Simulator v2.0: Advanced Phenomenological Coefficients Theory
Modularized implementation of the Theory of Phenomenological Coefficients.

## DOI Update for: https://doi.org/10.5281/zenodo.17619907

### Summary

This major update to the Neurophenomenological Simulator introduces comprehensive scientific validation mechanisms and implements the complete mathematical framework of the Theory of Phenomenological Coefficients. The new version includes rigorous experimental validation, advanced isomorphism analysis between neural and phenomenal domains, and enhanced computational robustness.

### Key New Features

#### 1. Advanced Scientific Validation
- **Isomorphism Analysis**: Implementation of Theorem 3 (Strict Mind-Brain Isomorphism Φ: ℋ → 𝒩)
- **Experimental Validator**: Statistical validation of gamma-coefficient correlation
- **Threshold Validation**: ROC curve analysis for phenomenological threshold γ_min ≈ 0.3
- **State Transition Analysis**: Hysteresis and transition dynamics

#### 2. Enhanced Computational Framework
- **Extended Simulator**: `SimuladorCompletoExtendido` integrates new components without modifying original code
- **Robust Statistical Analysis**: Handles edge cases and data variability
- **Automatic Scientific Reporting**: Generates comprehensive validation reports

#### 3. Improved User Interface
- **Dual Simulation Modes**: Basic and Advanced scientific analysis
- **Specialized Analysis Tabs**: Isomorphism, validation, state transitions, mind-brain mapping
- **Real-time Validation Metrics**: Live scientific validation during simulation

### Technical Specifications

#### New Classes Added:
- `IsomorfismoMenteCerebro`: Strict mind-brain isomorphism mapping
- `ValidadorExperimental`: Experimental validation methods
- `SimuladorCompletoExtendido`: Extended simulator with validation

#### Enhanced Scientific Capabilities:
- Pearson correlation and R² analysis for Γ-|c| isomorphism
- ROC curve analysis for threshold validation
- State transition latency and hysteresis analysis
- Entropy-based synchronization measures

## Installation
pip install .


### Usage

#### Basic Simulation:
```python
simulador = SimuladorCompleto(config)
resultados = simulador.simular_paradigma_experimental("validacion_gamma")

Advanced Scientific Analysis:
python

simulador_ext = SimuladorCompletoExtendido(config)
resultados = simulador_ext.simular_paradigma_con_validacion("validacion_gamma")
reporte = simulador_ext.generar_reporte_cientifico(resultados)

Citation
bibtex

@software{morelos_navidad_2024_17689427,
  author       = {Marco Antonio Morelos Navidad},
  title        = {Neurophenomenological Simulator v2.0: Advanced 
                 Phenomenological Coefficients Theory},
  month        = dec,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {2.0},
  doi          = {10.5281/zenodo.17689427},
  url          = {https://doi.org/10.5281/zenodo.17689427}
}

License

MIT License
Author

Marco Antonio Morelos Navidad
Universidad Autónoma Metropolitana
ORCID: 0009-0007-0083-5496


