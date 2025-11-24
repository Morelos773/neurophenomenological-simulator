import numpy as np
import numpy.typing as npt
from scipy import linalg
from typing import Dict, Optional, List
from .espacio import EspacioEstadosConscientes
from .configuracion import ConfiguracionNeurofisiologica

class CampoConscienteUnificado:
    def __init__(self, espacio_estados: EspacioEstadosConscientes, config: ConfiguracionNeurofisiologica):
        self.espacio = espacio_estados
        self.config = config
        self.operadores = self._inicializar_operadores()
        
    def _inicializar_operadores(self) -> Dict[str, npt.NDArray]:
        dim = self.espacio.dimension
        H_base = np.diag(np.random.normal(0, 0.1, dim))
        V_att = np.random.normal(0, 0.05, (dim, dim)) + 1j*np.random.normal(0, 0.05, (dim, dim))
        V_att = (V_att + V_att.conj().T) / 2
        V_mem = np.random.normal(0, 0.03, (dim, dim)) + 1j*np.random.normal(0, 0.03, (dim, dim))
        V_mem = (V_mem + V_mem.conj().T) / 2
        return {'H_base': H_base, 'V_att': V_att, 'V_mem': V_mem}
    
    def evolucion_temporal(self, estado: npt.NDArray, dt: float, inputs_externos: Optional[List] = None) -> npt.NDArray:
        H_total = (self.operadores['H_base'] + self.operadores['V_att'] + self.operadores['V_mem'])
        
        if inputs_externos:
            for V_ext in inputs_externos:
                H_total += V_ext
        
        hbar_eff = 0.1
        U = linalg.expm(-1j * H_total * dt / hbar_eff)
        ruido = np.random.normal(0, 0.01, estado.shape) + 1j*np.random.normal(0, 0.01, estado.shape)
        nuevo_estado = U @ estado + ruido * np.sqrt(dt)
        
        return self._aplicar_conservacion_recursos(nuevo_estado)
    
    def _aplicar_conservacion_recursos(self, estado: npt.NDArray) -> npt.NDArray:
        norma_actual = self.espacio.norma(estado)
        if norma_actual > self.config.capacidad_maxima:
            factor_normalizacion = self.config.capacidad_maxima / norma_actual
            return estado * factor_normalizacion
        return estado
