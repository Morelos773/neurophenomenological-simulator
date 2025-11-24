import numpy as np
import numpy.typing as npt
from scipy import signal
from typing import Optional
from .sistema import SistemaMicronodosNeurales

class CoeficienteFenomenologicoCompleto:
    def __init__(self, sistema_neural: SistemaMicronodosNeurales):
        self.sistema = sistema_neural
        self.parametros_experimentales = {
            'coherencia_gamma_umbral': 0.25,
            'firing_rate_maximo': 1.0,
            'banda_gamma': (30.0, 80.0)
        }
    
    def calcular_gamma(self, fases: npt.NDArray, tiempo: float) -> float:
        senal_compleja = np.exp(1j * fases)
        fase_instantanea = np.angle(signal.hilbert(np.real(senal_compleja)))
        orden_complejo = np.mean(np.exp(1j * fase_instantanea))
        gamma = np.abs(orden_complejo)
        
        if gamma < 0.25:
            return 0.0
        elif gamma > 0.9:
            return 1.0
        else:
            return gamma
    
    def calcular_amplitud(self, amplitudes: npt.NDArray) -> float:
        firing_rate_promedio = np.mean(amplitudes)
        firing_rate_maximo = self.parametros_experimentales['firing_rate_maximo']
        A_normalizada = firing_rate_promedio / firing_rate_maximo
        return np.clip(A_normalizada, 0.0, 1.0)
    
    def calcular_fase_relativa(self, fases: npt.NDArray) -> float:
        return np.angle(np.mean(np.exp(1j * fases)))
    
    def coeficiente_complejo(self, fases: npt.NDArray, amplitudes: npt.NDArray, tiempo: float) -> complex:
        gamma = self.calcular_gamma(fases, tiempo)
        A = self.calcular_amplitud(amplitudes)
        theta = self.calcular_fase_relativa(fases)
        return gamma * A * np.exp(1j * theta)
