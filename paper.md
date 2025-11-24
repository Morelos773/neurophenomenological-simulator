---
title: 'Neurophenomenological Simulator: A Mathematical Framework for Consciousness Studies'
tags:
  - Python
  - computational neuroscience
  - consciousness
  - mathematical modeling
  - gamma oscillations
  - phenomenological coefficients
authors:
  - name: Marco Antonio Morelos Navidad
    orcid: 0009-0007-0083-5496
    affiliation: 1
affiliations:
  - name: Universidad Autónoma Metropolitana
    index: 1
date: 19 December 2024
bibliography: paper.bib
---

# Summary

The Neurophenomenological Simulator implements a novel mathematical framework for consciousness studies, bridging phenomenological experience with neural dynamics through the Theory of Phenomenological Coefficients. This open-source software provides rigorous computational tools for modeling consciousness, validating theoretical predictions, and enabling reproducible research in consciousness studies.

# Statement of Need

Consciousness research faces a fundamental challenge: bridging the explanatory gap between first-person subjective experience and third-person neural observations. While numerous theoretical frameworks exist, few offer mathematically rigorous, computationally implementable models that can be empirically validated. The Neurophenomenological Simulator addresses this critical gap by providing:

- A **complete mathematical formalization** of conscious states
- **Computational implementation** of phenomenological principles  
- **Experimental validation** mechanisms for theoretical predictions
- **Reproducible framework** for consciousness modeling

Current computational approaches often focus on neural correlates without phenomenological grounding or remain purely theoretical without computational implementation. Our software uniquely integrates mathematical rigor, computational practicality, and empirical validation in a single framework.

# Mathematics and Theory

## Phenomenological Coefficients Theory

The simulator implements the complete mathematical framework:

### Axiom 1: Conscious State Space
Conscious states reside in a Hilbert space $\mathcal{H}$ of dimension $N$, with orthonormal basis vectors $\{|\psi_i\rangle\}$ representing elementary qualia.

### Axiom 2: Phenomenological Coefficients  
Each conscious state is characterized by time-dependent coefficients:
$$c_i(t) = \Gamma_i(t) \cdot A_i(t) \cdot e^{i\theta_i(t)}$$
where:
- $\Gamma_i(t)$: Gamma synchronization degree (0-1)
- $A_i(t)$: Normalized activation amplitude (0-1)  
- $\theta_i(t)$: Coherent relative phase

### Theorem 1: Unified Conscious Field
The global conscious experience emerges as:
$$|\Psi(t)\rangle = \sum_{i=1}^N c_i(t)|\psi_i\rangle$$

### Theorem 2: Resource Conservation
Conscious capacity is bounded:
$$\langle\Psi|\Psi\rangle \leq C_{\text{max}}$$

### Theorem 3: Mind-Brain Isomorphism
There exists a strict isomorphism $\Phi: \mathcal{H} \rightarrow \mathcal{N}$ between phenomenal states and neural patterns.

# Features

## Core Computational Capabilities

1. **Conscious State Space Management**
   - Hilbert space operations and transformations
   - Orthonormal basis generation and maintenance
   - Phenomenological distance calculations

2. **Neural System Simulation** 
   - Mesoscopic neural modeling (cortical columns)
   - Gamma oscillation dynamics (30-80 Hz)
   - Kuramoto-type phase synchronization
   - Anatomically realistic connectivity

3. **Phenomenological Coefficient Computation**
   - Real-time coefficient calculation
   - Gamma synchronization measurement
   - Activation amplitude normalization
   - Phase coherence analysis

4. **Unified Conscious Field Integration**
   - State superposition and evolution
   - Resource conservation enforcement
   - Temporal dynamics simulation

## Advanced Scientific Validation

- **Isomorphism Validation**: Statistical correlation between neural synchronization ($\Gamma$) and conscious presence ($|c|$)
- **Threshold Validation**: ROC analysis for consciousness threshold $\gamma_{\text{min}} \approx 0.3$
- **State Transition Analysis**: Hysteresis detection and transition dynamics
- **Automated Reporting**: Comprehensive scientific validation metrics

## Experimental Paradigms

The simulator includes four pre-configured experimental paradigms:

1. **Gamma Validation**: Testing gamma synchronization hypotheses
2. **General Anesthesia**: Modeling consciousness loss
3. **Microstimulation**: Focal neural stimulation effects  
4. **Baseline State**: Normal conscious functioning

# Implementation

## Architecture

The software employs a modular, object-oriented architecture:

class EspacioEstadosConscientes:      # Conscious state space
class SistemaMicronodosNeurales:      # Neural system
class CoeficienteFenomenologicoCompleto: # Phenomenological coefficients  
class CampoConscienteUnificado:       # Unified conscious field
class SimuladorCompletoExtendido:     # Main simulator with validation

Technical Specifications
    Language: Python 3.8+
    Dependencies: NumPy, SciPy, Streamlit, Plotly, pandas, scikit-learn
    License: MIT
    Platform: Cross-platform (Windows, macOS, Linux)
    Interface: Web-based (Streamlit) for accessibility

Validation and Testing

The software includes comprehensive testing:
    Unit tests for core mathematical operations
    Statistical validation of theoretical predictions
    Integration tests for full simulation pipelines
    Example-based testing with known outcomes

Availability
    Source Code: https://github.com/[username]/neurophenomenological-simulator
    Archived Version: https://doi.org/10.5281/zenodo.17689427
    Live Demo: Available via Streamlit interface
    Documentation: Comprehensive examples and API reference
    License: MIT Open Source

Acknowledgements

This work builds upon foundations in phenomenological philosophy, mathematical neuroscience, and consciousness studies. We acknowledge the theoretical traditions that informed this computational implementation.
References
---
