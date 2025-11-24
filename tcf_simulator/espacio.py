import numpy as np
import numpy.typing as npt
from scipy import linalg
from typing import Any
from .configuracion import ConfiguracionNeurofisiologica

class EspacioEstadosConscientes:
    def __init__(self, config: ConfiguracionNeurofisiologica):
        self.config = config
        self.dimension = config.N_dimension
        self.estados_base = self._generar_base_completa()
        self.distancias_fenomenologicas = self._calcular_distancias_fenomenologicas()
        
    def _generar_base_completa(self) -> npt.NDArray[np.complex128]:
        base = np.eye(self.dimension, dtype=np.complex128)
        theta = np.random.uniform(0, 2*np.pi, (self.dimension, self.dimension))
        U = linalg.expm(1j * (theta + theta.T) / 2)
        return U @ base
    
    def _calcular_distancias_fenomenologicas(self) -> npt.NDArray[np.float64]:
        d_matrix = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                d_matrix[i, j] = abs(i - j) / self.dimension
        return d_matrix
    
    def producto_interno(self, estado1: npt.NDArray, estado2: npt.NDArray) -> complex:
        return np.vdot(estado1, estado2)
    
    def norma(self, estado: npt.NDArray) -> float:
        return np.sqrt(np.real(self.producto_interno(estado, estado)))
    
    def estado_qualia_elemental(self, indice: int) -> npt.NDArray:
        return self.estados_base[indice]
