---
title: "Neurophenomenological Simulator: A Computational Framework Implementing the Theory of Phenomenological Coefficients"
tags:
  - Python
  - computational neuroscience
  - consciousness modeling
  - gamma oscillations
  - neural simulation
  - phenomenological coefficients
authors:
  - name: Marco Antonio Morelos Navidad
    orcid: 0009-0007-0083-5496
    affiliation: "1"
affiliations:
  - name: Universidad Autónoma Metropolitana, Unidad Lerma
    index: 1
date: 2025-01-15
bibliography: paper.bib
---

# Summary

The Neurophenomenomenological Simulator implements a computational framework for modeling the relationship between neural dynamics and phenomenological experience based on the Theory of Phenomenological Coefficients (TPC). The software provides tools for simulating gamma-band synchronization, neural oscillatory interactions, and the temporal evolution of a phenomenological state represented in a Hilbert space. The simulator enables reproducible and quantitative exploration of consciousness-related hypotheses and supports experimental paradigms including baseline activity, anesthesia, gamma synchronization, and microstimulation.

# Statement of Need

Computational consciousness research lacks tools that directly integrate mathematical phenomenology with neural simulation. Existing frameworks typically emphasize neural correlates without providing a mechanism for modeling phenomenological magnitudes. The Neurophenomenological Simulator fills this gap by offering:

- A mathematically well-defined implementation of the TPC.
- A modular architecture for simulating mesoscopic neural systems.
- Real-time computation of phenomenological coefficients.
- Reproducible pipelines for validating theoretical predictions.

This tool is intended for researchers in computational neuroscience, cognitive science, and theoretical consciousness studies who require a programmable and extensible platform for testing hypotheses relating phenomenology and neural dynamics.

# Mathematical Framework

The simulator operationalizes the TPC, where phenomenological coefficients are defined as:

\[
c_i(t) = \Gamma_i(t) A_i(t) e^{i\theta_i(t)}
\]

with \(\Gamma_i\) representing gamma synchronization, \(A_i\) oscillatory amplitude, and \(\theta_i\) the phase component.  
The global phenomenological state is described by:

\[
|\Psi(t)\rangle = \sum_{i=1}^{N} c_i(t)|\psi_i\rangle.
\]

This formulation enables quantitative predictions of conscious–unconscious transitions, synchronization thresholds, and dynamical stability.

# Functionality

The simulator provides the following capabilities:

- Mesoscopic neural modeling via coupled oscillators.
- Anatomically inspired connectivity.
- Real-time computation of synchronization, amplitude, and phase parameters.
- Construction and evolution of a high-dimensional conscious state.
- Built-in experimental paradigms: baseline, gamma enhancement, anesthesia, and microstimulation.
- Web interface developed in Streamlit for interactive exploration.
- Automatic validation metrics comparing theoretical expectations and simulated behavior.

# State of the Field

Current computational frameworks in neuroscience emphasize neural signal processing, network dynamics, or correlational analyses but rarely incorporate explicit phenomenological quantities. The Neurophenomenological Simulator contributes a novel computational approach, enabling direct numerical manipulation of mathematically defined phenomenological states, offering compatibility with empirical neural dynamics models.

# Example Usage

```python
from tcf_simulator import CompleteExtendedSimulator, NeurophysiologicalConfiguration

config = NeurophysiologicalConfiguration(N_dimension=1000)
sim = CompleteExtendedSimulator(config)

results = sim.simulate_paradigm_with_validation("baseline")
sim.visualize_results(results)
```

# Acknowledgements

The software was developed with support from the Universidad Autónoma Metropolitana, Unidad Lerma.

# References
