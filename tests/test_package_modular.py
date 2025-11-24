import pytest
import numpy as np
from tcf_simulator import (
    ConfiguracionNeurofisiologica,
    EspacioEstadosConscientes,
    SistemaMicronodosNeurales,
    CoeficienteFenomenologicoCompleto
)

def test_configuration():
    cfg = ConfiguracionNeurofisiologica(N_dimension=100, N_micronodos=50)
    assert cfg.N_dimension == 100
    assert cfg.N_micronodos == 50

def test_espacio_and_norma():
    cfg = ConfiguracionNeurofisiologica(N_dimension=50)
    espacio = EspacioEstadosConscientes(cfg)
    estado = espacio.estado_qualia_elemental(0)
    assert abs(espacio.norma(estado) - 1.0) <= 1e-6

def test_sistema_and_coeficiente():
    cfg = ConfiguracionNeurofisiologica(N_dimension=100, N_micronodos=30)
    sistema = SistemaMicronodosNeurales(cfg)
    coef = CoeficienteFenomenologicoCompleto(sistema)
    fases = np.random.uniform(0, 2*np.pi, cfg.N_micronodos)
    amps = np.random.uniform(0.1, 0.3, cfg.N_micronodos)
    c = coef.coeficiente_complejo(fases, amps, tiempo=0.0)
    assert isinstance(c, complex)
    assert 0.0 <= abs(c) <= 1.0
