import numpy as np
import numpy.typing as npt
from typing import Dict, List
from .espacio import EspacioEstadosConscientes
from .sistema import SistemaMicronodosNeurales
from scipy import signal

class IsomorfismoMenteCerebro:
    """
    Implementa el Teorema 3: Isomorfismo Estricto Mente-Cerebro Φ: ℋ → 𝒩
    """
    
    def __init__(self, espacio_estados: EspacioEstadosConscientes, 
                 sistema_neural: SistemaMicronodosNeurales):
        self.espacio = espacio_estados
        self.sistema = sistema_neural
        self.mapeo_estados = self._inicializar_mapeo_estados()
        
    def _inicializar_mapeo_estados(self) -> Dict[int, List[int]]:
        mapeo = {}
        micronodos_por_estado = max(1, self.sistema.config.N_micronodos // self.espacio.dimension)
        
        for i in range(min(100, self.espacio.dimension)):
            inicio = (i * micronodos_por_estado) % self.sistema.config.N_micronodos
            fin = inicio + micronodos_por_estado
            mapeo[i] = list(range(inicio, min(fin, self.sistema.config.N_micronodos)))
            
        return mapeo
    
    def estado_neural_correspondiente(self, estado_consciente: npt.NDArray, 
                                    tiempo: float) -> Dict[str, npt.NDArray]:
        coeficientes_dominantes = np.argsort(np.abs(estado_consciente))[-10:]
        fases, amplitudes = self.sistema.micronodos['fases'], self.sistema.micronodos['amplitudes']
        
        estado_neural = {
            'coeficientes_dominantes': coeficientes_dominantes,
            'gamma_promedio': self._calcular_gamma_agregado(fases, coeficientes_dominantes),
            'amplitud_promedio': self._calcular_amplitud_agregada(amplitudes, coeficientes_dominantes),
            'fase_coherente': self._calcular_fase_coherente(fases, coeficientes_dominantes),
            'entropia_sincronizacion': self._calcular_entropia_sincronizacion(fases),
            'timestamp': tiempo
        }
        
        return estado_neural
    
    def _calcular_gamma_agregado(self, fases: npt.NDArray, 
                               estados_activos: npt.NDArray) -> float:
        if len(estados_activos) == 0:
            return 0.0
            
        gamma_estados = []
        for estado_idx in estados_activos:
            if estado_idx in self.mapeo_estados:
                micronodos_estado = self.mapeo_estados[estado_idx]
                if micronodos_estado:
                    fases_estado = fases[micronodos_estado]
                    senal_compleja = np.exp(1j * fases_estado)
                    fase_instantanea = np.angle(signal.hilbert(np.real(senal_compleja)))
                    orden_complejo = np.mean(np.exp(1j * fase_instantanea))
                    gamma_estados.append(np.abs(orden_complejo))
        
        return np.mean(gamma_estados) if gamma_estados else 0.0
    
    def _calcular_amplitud_agregada(self, amplitudes: npt.NDArray, 
                                  estados_activos: npt.NDArray) -> float:
        if len(estados_activos) == 0:
            return 0.0
            
        amplitudes_estados = []
        for estado_idx in estados_activos:
            if estado_idx in self.mapeo_estados:
                micronodos_estado = self.mapeo_estados[estado_idx]
                if micronodos_estado:
                    amps_estado = amplitudes[micronodos_estado]
                    amplitudes_estados.append(np.mean(amps_estado))
        
        return np.mean(amplitudes_estados) if amplitudes_estados else 0.0
    
    def _calcular_fase_coherente(self, fases: npt.NDArray, 
                               estados_activos: npt.NDArray) -> float:
        if len(estados_activos) == 0:
            return 0.0
            
        fases_agregadas = []
        for estado_idx in estados_activos:
            if estado_idx in self.mapeo_estados:
                micronodos_estado = self.mapeo_estados[estado_idx]
                if micronodos_estado:
                    fases_estado = fases[micronodos_estado]
                    fases_agregadas.extend(fases_estado)
        
        return np.angle(np.mean(np.exp(1j * np.array(fases_agregadas)))) if fases_agregadas else 0.0
    
    def _calcular_entropia_sincronizacion(self, fases: npt.NDArray) -> float:
        hist, _ = np.histogram(fases, bins=20, range=(0, 2*np.pi), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist)) / np.log(len(hist)) if len(fases) > 1 else 0.0
