# Installation and Setup

## Requirements

### Python Version
- Python 3.8 or higher

### Required Packages
```bash
numpy>=1.21.0
matplotlib>=3.5.0
streamlit>=1.28.0
scipy>=1.7.0
plotly>=5.0.0
pandas>=1.3.0
scikit-learn>=1.0.0

Installation Command
bash

pip install numpy matplotlib streamlit scipy plotly pandas scikit-learn

Quick Start
1. Download the Simulator
bash

git clone [repository-url]
cd neurophenomenological-simulator

2. Run Basic Simulation
bash

streamlit run main_actual.py

3. Access the Interface

    Open web browser to http://localhost:8501

    Configure parameters in sidebar

    Click "Execute Simulation"

Usage Examples
Basic Scientific Analysis
python

from main_actual import ConfiguracionNeurofisiologica, SimuladorCompleto

config = ConfiguracionNeurofisiologica(
    N_dimension=1000,
    N_micronodos=100,
    frecuencia_gamma=(30, 80)
)

simulador = SimuladorCompleto(config)
resultados = simulador.simular_paradigma_experimental("validacion_gamma")

Advanced Validation Analysis
python

from main_actual import SimuladorCompletoExtendido

simulador_ext = SimuladorCompletoExtendido(config)
resultados = simulador_ext.simular_paradigma_con_validacion("validacion_gamma")
reporte = simulador_ext.generar_reporte_cientifico(resultados)
print(reporte)

Experimental Paradigms
Available Paradigms

    validacion_gamma: Gamma synchronization validation

    anestesia_general: General anesthesia simulation

    microestimulacion: Microstimulation effects

    basal: Baseline conscious state

Configuration Parameters

    Dimension N: 100, 500, 1000

    Micronodes: 50-200

    Gamma frequency: 30-80 Hz

    Phenomenological threshold: 0.1-0.5

    Capacity maximum: 0.5-2.0

Troubleshooting
Common Issues

    Import Errors

        Ensure all required packages are installed

        Check Python version compatibility

    Memory Issues

        Reduce dimension N for large simulations

        Decrease number of micronodes

    Visualization Problems

        Clear browser cache

        Check plotly version compatibility

Performance Tips

    Use smaller dimensions for quick testing

    Reduce simulation duration for rapid iterations

    Close other memory-intensive applications