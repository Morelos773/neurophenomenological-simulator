import numpy as np
import numpy.typing as npt
from typing import Dict, List, Any
from .configuracion import ConfiguracionNeurofisiologica
from sklearn.metrics import roc_curve, auc
from scipy import stats

class ValidadorExperimental:
    """
    Implementa métodos de validación experimental del marco teórico.
    """
    
    def __init__(self, config: ConfiguracionNeurofisiologica):
        self.config = config
        self.parametros_referencia = self._inicializar_parametros_referencia()
    
    def _inicializar_parametros_referencia(self) -> Dict[str, tuple]:
        return {
            'gamma_consciente': (0.6, 0.9),
            'gamma_inconsciente': (0.0, 0.25),
            'umbral_fenomenologico': (0.28, 0.32),
            'amplitud_umbral': (0.22, 0.25),
            'correlacion_gamma_coeficiente': (0.7, 1.0)
        }
    
    def validar_isomorfismo_gamma_coeficiente(self, gamma_values: List[float], 
                                            coeficiente_values: List[float]) -> Dict[str, float]:
        gamma_arr = np.array(gamma_values)
        coeficiente_arr = np.array(coeficiente_values)
        
        # Verificar si todos los valores de gamma son idénticos
        if np.all(gamma_arr == gamma_arr[0]):
            return {
                'correlacion_pearson': 0.0,
                'r_cuadrado': 0.0,
                'p_value': 1.0,
                'pendiente_regresion': 0.0,
                'error_cuadratico_medio': np.var(coeficiente_arr),
                'isomorfismo_valido': False,
                'advertencia': 'Todos los valores de Gamma son idénticos'
            }
        
        # Verificar si hay suficiente variabilidad para calcular correlación
        if len(gamma_arr) < 2 or np.std(gamma_arr) < 1e-10:
            return {
                'correlacion_pearson': 0.0,
                'r_cuadrado': 0.0,
                'p_value': 1.0,
                'pendiente_regresion': 0.0,
                'error_cuadratico_medio': np.var(coeficiente_arr),
                'isomorfismo_valido': False,
                'advertencia': 'Variabilidad insuficiente en Gamma'
            }
        
        try:
            correlacion = np.corrcoef(gamma_arr, coeficiente_arr)[0, 1]
            pendiente, intercepto, r_value, p_value, std_err = stats.linregress(gamma_arr, coeficiente_arr)
            
            predicciones = pendiente * gamma_arr + intercepto
            mse = np.mean((coeficiente_arr - predicciones) ** 2)
            
            return {
                'correlacion_pearson': correlacion,
                'r_cuadrado': r_value ** 2,
                'p_value': p_value,
                'pendiente_regresion': pendiente,
                'error_cuadratico_medio': mse,
                'isomorfismo_valido': correlacion > 0.7 and p_value < 0.05,
                'advertencia': None
            }
        except Exception as e:
            return {
                'correlacion_pearson': 0.0,
                'r_cuadrado': 0.0,
                'p_value': 1.0,
                'pendiente_regresion': 0.0,
                'error_cuadratico_medio': np.var(coeficiente_arr),
                'isomorfismo_valido': False,
                'advertencia': f'Error en cálculo: {str(e)}'
            }
    
    def validar_umbral_fenomenologico(self, coeficiente_values: List[float], 
                                    estados_consciente: List[bool]) -> Dict[str, float]:
        coef_arr = np.array(coeficiente_values)
        consciente_arr = np.array(estados_consciente)
        
        if len(np.unique(consciente_arr)) < 2:
            return {'error': 'Se necesitan ambos estados (consciente/inconsciente)'}
        
        fpr, tpr, thresholds = roc_curve(consciente_arr, coef_arr)
        roc_auc = auc(fpr, tpr)
        
        distancias = np.sqrt(fpr**2 + (1-tpr)**2)
        umbral_optimo_idx = np.argmin(distancias)
        umbral_optimo = thresholds[umbral_optimo_idx]
        
        predicciones_teoricas = coef_arr > 0.3
        precision_teorica = np.mean(predicciones_teoricas == consciente_arr)
        
        return {
            'umbral_optimo_roc': umbral_optimo,
            'area_bajo_curva': roc_auc,
            'precision_umbral_teorico': precision_teorica,
            'umbral_valido': abs(umbral_optimo - 0.3) < 0.05 and precision_teorica > 0.85
        }
    
    def analizar_transiciones_estado(self, coeficiente_values: List[float], 
                                   tiempo_values: List[float]) -> Dict[str, Any]:
        coef_arr = np.array(coeficiente_values)
        tiempo_arr = np.array(tiempo_values)
        
        estados = coef_arr > 0.3
        transiciones = np.diff(estados.astype(int))
        
        transiciones_positivas = np.where(transiciones > 0)[0]
        transiciones_negativas = np.where(transiciones < 0)[0]
        
        caracteristicas = {
            'n_transiciones_positivas': len(transiciones_positivas),
            'n_transiciones_negativas': len(transiciones_negativas),
            'duracion_estados_conscientes': self._calcular_duracion_estados(estados, tiempo_arr, True),
            'duracion_estados_inconscientes': self._calcular_duracion_estados(estados, tiempo_arr, False),
            'latencia_transiciones': self._calcular_latencia_transiciones(transiciones_positivas, tiempo_arr),
            'histéresis_umbral': self._analizar_histéresis(coef_arr, transiciones_positivas, transiciones_negativas)
        }
        
        return caracteristicas
    
    def _calcular_duracion_estados(self, estados: npt.NDArray, 
                                 tiempos: npt.NDArray, estado_objetivo: bool) -> list:
        cambios = np.diff(estados.astype(int))
        inicio_episodios = np.where(cambios == (1 if estado_objetivo else -1))[0] + 1
        fin_episodios = np.where(cambios == (-1 if estado_objetivo else 1))[0] + 1
        
        if estado_objetivo and estados[0]:
            inicio_episodios = np.insert(inicio_episodios, 0, 0)
        if not estado_objetivo and not estados[0]:
            inicio_episodios = np.insert(inicio_episodios, 0, 0)
        if estado_objetivo and estados[-1]:
            fin_episodios = np.append(fin_episodios, len(estados))
        if not estado_objetivo and not estados[-1]:
            fin_episodios = np.append(fin_episodios, len(estados))
        
        duraciones = []
        for inicio, fin in zip(inicio_episodios, fin_episodios):
            if fin > inicio:
                duraciones.append(tiempos[fin-1] - tiempos[inicio])
        
        return duraciones
    
    def _calcular_latencia_transiciones(self, indices_transicion: npt.NDArray, 
                                      tiempos: npt.NDArray) -> float:
        if len(indices_transicion) < 2:
            return 0.0
        intervalos = np.diff(tiempos[indices_transicion])
        return float(np.mean(intervalos)) if len(intervalos) > 0 else 0.0
    
    def _analizar_histéresis(self, coeficientes: npt.NDArray, 
                           trans_pos: npt.NDArray, trans_neg: npt.NDArray) -> dict:
        if len(trans_pos) == 0 or len(trans_neg) == 0:
            return {'histéresis_detectada': False, 'magnitud_histéresis': 0.0}
        
        valores_trans_pos = coeficientes[trans_pos]
        valores_trans_neg = coeficientes[trans_neg]
        
        histéresis = np.mean(valores_trans_pos) - np.mean(valores_trans_neg)
        
        return {
            'histéresis_detectada': abs(histéresis) > 0.02,
            'magnitud_histéresis': float(histéresis),
            'umbral_activacion': float(np.mean(valores_trans_pos)) if len(valores_trans_pos)>0 else 0.0,
            'umbral_desactivacion': float(np.mean(valores_trans_neg)) if len(valores_trans_neg)>0 else 0.0
        }
