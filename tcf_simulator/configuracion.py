from dataclasses import dataclass
from typing import Tuple

@dataclass
class ConfiguracionNeurofisiologica:
    N_dimension: int = 10**3
    N_micronodos: int = 100
    resolucion_temporal: float = 0.01
    frecuencia_gamma: Tuple[float, float] = (30.0, 80.0)
    umbral_fenomenologico: float = 0.3
    capacidad_maxima: float = 1.0
    tamano_columna_cortical: float = 0.5
    neuronas_por_micronodo: int = 10000
    conectividad_promedio: float = 0.6
