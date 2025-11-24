import numpy as np
import numpy.typing as npt
from typing import Dict
from .configuracion import ConfiguracionNeurofisiologica

class SistemaMicronodosNeurales:
    def __init__(self, config: ConfiguracionNeurofisiologica):
        self.config = config
        self.micronodos = self._inicializar_micronodos()
        self.matriz_conectividad = self._generar_conectividad_anatómica()
        
    def _inicializar_micronodos(self) -> Dict[str, npt.NDArray]:
        return {
            'fases': np.random.uniform(0, 2*np.pi, self.config.N_micronodos),
            'amplitudes': np.random.uniform(0.1, 0.2, self.config.N_micronodos),
            'frecuencias_naturales': np.random.uniform(
                self.config.frecuencia_gamma[0], 
                self.config.frecuencia_gamma[1], 
                self.config.N_micronodos
            ),
            'umbrales_activacion': np.random.uniform(0.15, 0.25, self.config.N_micronodos),
            'posiciones_corticales': np.random.uniform(0, 1, (self.config.N_micronodos, 3))
        }
    
    def _generar_conectividad_anatómica(self) -> npt.NDArray:
        posiciones = self.micronodos['posiciones_corticales']
        distancias = np.linalg.norm(posiciones[:, np.newaxis] - posiciones, axis=2)
        return np.exp(-distancias**2 / 0.1) * self.config.conectividad_promedio
    
    def evolucionar_sistema_neural(self, t: float, condiciones: Dict) -> Dict:
        fases = self.micronodos['fases'].copy()
        amplitudes = self.micronodos['amplitudes'].copy()
        frecuencias = self.micronodos['frecuencias_naturales']
        K = condiciones.get('acoplamiento', 0.8)
        
        for i in range(self.config.N_micronodos):
            suma_acoplamiento = 0
            for j in range(self.config.N_micronodos):
                if i != j:
                    suma_acoplamiento += self.matriz_conectividad[i,j] * np.sin(fases[j] - fases[i])
            fases[i] += self.config.resolucion_temporal * (frecuencias[i] + (K/self.config.N_micronodos) * suma_acoplamiento)
        
        ruido = np.random.normal(0, 0.01, self.config.N_micronodos)
        amplitudes += self.config.resolucion_temporal * (-0.1 * (amplitudes - 0.15) + 0.05 * ruido)
        amplitudes = np.clip(amplitudes, 0, 1)
        
        if 'estimulo_externo' in condiciones:
            tiempo_estimulo = condiciones.get('tiempo_estimulo', 3.0)
            duracion_estimulo = condiciones.get('duracion_estimulo', 1.0)
            if tiempo_estimulo <= t <= tiempo_estimulo + duracion_estimulo:
                amplitudes += condiciones['estimulo_externo']
                amplitudes = np.clip(amplitudes, 0, 1)
        
        self.micronodos['fases'] = fases
        self.micronodos['amplitudes'] = amplitudes
        
        return fases, amplitudes
