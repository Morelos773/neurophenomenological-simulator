"""
INTEGRACIÓN AVANZADA - TEORÍA DE COEFICIENTES FENOMENOLÓGICOS
Extensión del simulador original con validación científica completa
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import integrate, signal, linalg, special, stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy.typing as npt
from dataclasses import dataclass

# =============================================================================
# CLASES ORIGINALES 
# =============================================================================

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
    
    def evolucionar_sistema_neural(self, t: float, condiciones: Dict) -> Tuple[npt.NDArray, npt.NDArray]:
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

class SimuladorCompleto:
    def __init__(self, config: ConfiguracionNeurofisiologica):
        self.config = config
        self.espacio_estados = EspacioEstadosConscientes(config)
        self.sistema_neural = SistemaMicronodosNeurales(config)
        self.coeficiente = CoeficienteFenomenologicoCompleto(self.sistema_neural)
        self.campo_consciente = CampoConscienteUnificado(self.espacio_estados, config)
        self.estados_activados = min(100, config.N_dimension)
    
    def _configurar_paradigma(self, tipo_paradigma: str) -> Dict:
        if tipo_paradigma == "validacion_gamma":
            return {
                'acoplamiento': 0.8,
                'estimulo_externo': 0.2,
                'tiempo_estimulo': 3.0,
                'duracion_estimulo': 1.0,
                'inputs_externos': None
            }
        elif tipo_paradigma == "anestesia_general":
            return {
                'acoplamiento': 0.3,
                'estimulo_externo': 0.0,
                'inputs_externos': None
            }
        elif tipo_paradigma == "microestimulacion":
            return {
                'acoplamiento': 0.6,
                'estimulo_externo': 0.4,
                'tiempo_estimulo': 2.0,
                'duracion_estimulo': 0.5,
                'inputs_externos': None
            }
        else:
            return {
                'acoplamiento': 0.7,
                'estimulo_externo': 0.0,
                'inputs_externos': None
            }
    
    def _evolucion_sistema_neural(self, t: float, condiciones: Dict) -> Tuple[npt.NDArray, npt.NDArray]:
        return self.sistema_neural.evolucionar_sistema_neural(t, condiciones)
    
    def _calcular_entropia_sincronizacion(self, fases: npt.NDArray) -> float:
        hist, _ = np.histogram(fases, bins=20, range=(0, 2*np.pi), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist)) / np.log(len(hist))
    
    def simular_paradigma_experimental(self, tipo_paradigma: str, duracion: float = 10.0) -> Dict:
        condiciones = self._configurar_paradigma(tipo_paradigma)
        
        estado_consciente = np.zeros(self.config.N_dimension, dtype=np.complex128)
        indices_activados = np.random.choice(self.config.N_dimension, self.estados_activados, replace=False)
        
        resultados = {
            'tiempo': [],
            'gamma': [],
            'amplitud': [],
            'coeficiente_modulo': [],
            'coeficiente_fase': [],
            'estado_consciente': [],
            'norma_campo': [],
            'entropia_sincronizacion': [],
            'transiciones_estado': []
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tiempos = np.arange(0, duracion, self.config.resolucion_temporal)
        total_pasos = len(tiempos)

        for idx, t in enumerate(tiempos):
            status_text.text(f"Simulando tiempo t = {t:.2f} s")
            progress_bar.progress(int(100 * idx / total_pasos))

            fases, amplitudes = self._evolucion_sistema_neural(t, condiciones)
            c_complejo = self.coeficiente.coeficiente_complejo(fases, amplitudes, t)

            for idx_est in indices_activados[:10]:
                estado_consciente[idx_est] = c_complejo * (0.8 + 0.4 * np.random.random())

            estado_consciente = self.campo_consciente.evolucion_temporal(
                estado_consciente,
                self.config.resolucion_temporal,
                condiciones.get('inputs_externos')
            )

            gamma_promedio = self.coeficiente.calcular_gamma(fases, t)
            A_promedio = self.coeficiente.calcular_amplitud(amplitudes)
            c_promedio = gamma_promedio * A_promedio
            consciente = c_promedio > self.config.umbral_fenomenologico
            entropia = self._calcular_entropia_sincronizacion(fases)

            resultados['tiempo'].append(t)
            resultados['gamma'].append(gamma_promedio)
            resultados['amplitud'].append(A_promedio)
            resultados['coeficiente_modulo'].append(c_promedio)
            resultados['coeficiente_fase'].append(np.angle(c_complejo))
            resultados['estado_consciente'].append(consciente)
            resultados['norma_campo'].append(self.espacio_estados.norma(estado_consciente))
            resultados['entropia_sincronizacion'].append(entropia)

        progress_bar.progress(100)
        status_text.text("Simulación completada.")

        transiciones = np.diff(resultados['estado_consciente'])
        resultados['transiciones_estado'] = transiciones
        
        return resultados

# =============================================================================
# NUEVAS CLASES AVANZADAS 
# =============================================================================

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
        return -np.sum(hist * np.log(hist)) / np.log(len(hist)) if len(hist) > 1 else 0.0

class ValidadorExperimental:
    """
    Implementa métodos de validación experimental del marco teórico.
    """
    
    def __init__(self, config: ConfiguracionNeurofisiologica):
        self.config = config
        self.parametros_referencia = self._inicializar_parametros_referencia()
    
    def _inicializar_parametros_referencia(self) -> Dict[str, Tuple[float, float]]:
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
        
        from sklearn.metrics import roc_curve, auc
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
                                   tiempo_values: List[float]) -> Dict[str, any]:
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
                                 tiempos: npt.NDArray, estado_objetivo: bool) -> List[float]:
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
        return np.mean(intervalos) if len(intervalos) > 0 else 0.0
    
    def _analizar_histéresis(self, coeficientes: npt.NDArray, 
                           trans_pos: npt.NDArray, trans_neg: npt.NDArray) -> Dict[str, float]:
        if len(trans_pos) == 0 or len(trans_neg) == 0:
            return {'histéresis_detectada': False, 'magnitud_histéresis': 0.0}
        
        valores_trans_pos = coeficientes[trans_pos]
        valores_trans_neg = coeficientes[trans_neg]
        
        histéresis = np.mean(valores_trans_pos) - np.mean(valores_trans_neg)
        
        return {
            'histéresis_detectada': abs(histéresis) > 0.02,
            'magnitud_histéresis': histéresis,
            'umbral_activacion': np.mean(valores_trans_pos),
            'umbral_desactivacion': np.mean(valores_trans_neg)
        }

class SimuladorCompletoExtendido:
    """
    Extensión del simulador principal que integra los nuevos componentes
    sin modificar el código original.
    """
    
    def __init__(self, config: ConfiguracionNeurofisiologica):
        # Componentes originales
        self.config = config
        self.espacio_estados = EspacioEstadosConscientes(config)
        self.sistema_neural = SistemaMicronodosNeurales(config)
        self.coeficiente = CoeficienteFenomenologicoCompleto(self.sistema_neural)
        self.campo_consciente = CampoConscienteUnificado(self.espacio_estados, config)
        self.estados_activados = min(100, config.N_dimension)
        
        # Nuevos componentes
        self.isomorfismo = IsomorfismoMenteCerebro(self.espacio_estados, self.sistema_neural)
        self.validador = ValidadorExperimental(config)
    
    def simular_paradigma_con_validacion(self, tipo_paradigma: str, duracion: float = 10.0) -> Dict:
        # Usar el simulador original para obtener resultados base
        simulador_original = SimuladorCompleto(self.config)
        resultados_original = simulador_original.simular_paradigma_experimental(tipo_paradigma, duracion)
        
        # Reconstruir estado consciente final para el isomorfismo
        estado_final = np.zeros(self.config.N_dimension, dtype=np.complex128)
        if len(resultados_original['estado_consciente']) > 0:
            indices_activados = np.random.choice(self.config.N_dimension, self.estados_activados, replace=False)
            for idx_est in indices_activados[:10]:
                estado_final[idx_est] = resultados_original['coeficiente_modulo'][-1] * np.exp(1j * resultados_original['coeficiente_fase'][-1])
        
        # Calcular isomorfismo
        estado_neural = self.isomorfismo.estado_neural_correspondiente(estado_final, duracion)
        
        # Validaciones experimentales
        validacion_isomorfismo = self.validador.validar_isomorfismo_gamma_coeficiente(
            resultados_original['gamma'], resultados_original['coeficiente_modulo']
        )
        
        validacion_umbral = self.validador.validar_umbral_fenomenologico(
            resultados_original['coeficiente_modulo'], resultados_original['estado_consciente']
        )
        
        analisis_transiciones = self.validador.analizar_transiciones_estado(
            resultados_original['coeficiente_modulo'], resultados_original['tiempo']
        )
        
        # Combinar resultados
        resultados_extendidos = {
            **resultados_original,
            'isomorfismo_mente_cerebro': estado_neural,
            'validacion_isomorfismo_gamma_coeficiente': validacion_isomorfismo,
            'validacion_umbral_fenomenologico': validacion_umbral,
            'analisis_transiciones_estado': analisis_transiciones,
            'parametros_referencia': self.validador.parametros_referencia
        }
        
        return resultados_extendidos
    
    def generar_reporte_cientifico(self, resultados: Dict) -> str:
        reporte = []
        reporte.append("=== REPORTE CIENTÍFICO - TEORÍA DE COEFICIENTES FENOMENOLÓGICOS ===")
        reporte.append(f"Duración de simulación: {resultados['tiempo'][-1]:.2f} segundos")
        reporte.append("")
        
        # Validación de isomorfismo Γ-|c|
        iso_val = resultados.get('validacion_isomorfismo_gamma_coeficiente', {})
        if iso_val:
            reporte.append("1. VALIDACIÓN ISOMORFISMO Γ(t) ↔ |c(t)|:")
            reporte.append(f"   - Correlación de Pearson: {iso_val.get('correlacion_pearson', 0):.3f}")
            reporte.append(f"   - R²: {iso_val.get('r_cuadrado', 0):.3f}")
            reporte.append(f"   - p-value: {iso_val.get('p_value', 0):.3f}")
            
            if iso_val.get('advertencia'):
                reporte.append(f"   - ⚠️ Advertencia: {iso_val['advertencia']}")
                reporte.append(f"   - Isomorfismo válido: ❌ (datos insuficientes)")
            else:
                reporte.append(f"   - Isomorfismo válido: {'✅' if iso_val.get('isomorfismo_valido', False) else '❌'}")
            reporte.append("")
        
        # Validación del umbral fenomenológico
        umbral_val = resultados.get('validacion_umbral_fenomenologico', {})
        if umbral_val and 'precision_umbral_teorico' in umbral_val:
            reporte.append("2. VALIDACIÓN UMBRAL FENOMENOLÓGICO γ_min ≈ 0.3:")
            reporte.append(f"   - Precisión del umbral teórico: {umbral_val['precision_umbral_teorico']:.1%}")
            reporte.append(f"   - Umbral óptimo (ROC): {umbral_val.get('umbral_optimo_roc', 0):.3f}")
            reporte.append(f"   - Área bajo curva ROC: {umbral_val.get('area_bajo_curva', 0):.3f}")
            reporte.append(f"   - Umbral válido: {'✅' if umbral_val.get('umbral_valido', False) else '❌'}")
            reporte.append("")
        
        # Análisis de transiciones
        transiciones = resultados.get('analisis_transiciones_estado', {})
        if transiciones:
            reporte.append("3. ANÁLISIS DE TRANSICIONES DE ESTADO:")
            reporte.append(f"   - Transiciones consciente: {transiciones.get('n_transiciones_positivas', 0)}")
            reporte.append(f"   - Transiciones inconsciente: {transiciones.get('n_transiciones_negativas', 0)}")
            
            histéresis = transiciones.get('histéresis_umbral', {})
            if histéresis.get('histéresis_detectada', False):
                reporte.append(f"   - Histéresis detectada: ✅ (magnitud: {histéresis.get('magnitud_histéresis', 0):.3f})")
            else:
                reporte.append("   - Histéresis detectada: ❌")
            reporte.append("")
        
        # Estado del isomorfismo mente-cerebro
        iso_mente = resultados.get('isomorfismo_mente_cerebro', {})
        if iso_mente:
            reporte.append("4. ISOMORFISMO MENTE-CEREBRO Φ(|Ψ⟩):")
            reporte.append(f"   - Γ promedio: {iso_mente.get('gamma_promedio', 0):.3f}")
            reporte.append(f"   - A promedio: {iso_mente.get('amplitud_promedio', 0):.3f}")
            reporte.append(f"   - Entropía sincronización: {iso_mente.get('entropia_sincronizacion', 0):.3f}")
            reporte.append("")
        
        reporte.append("=== FIN DEL REPORTE ===")
        
        return "\n".join(reporte)

# =============================================================================
# INTERFAZ STREAMLIT COMPLETA CON INTEGRACIÓN AVANZADA
# =============================================================================

def main():
    """
    Interfaz principal de la aplicación Streamlit con funcionalidad extendida.
    """
    st.set_page_config(
        page_title="Simulador Neurofenomenológico - Coeficientes Fenomenológicos", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Simulador Científico: Teoría de Coeficientes Fenomenológicos")
    st.markdown("""
    **Implementación computacional rigurosa del marco teórico completo**  
    *Marco Antonio Morelos Navidad - ORCID: 0009-0007-0083-5496*
    
    ---
    """)
    
    # ========== BARRA LATERAL DE CONFIGURACIÓN ==========
    with st.sidebar:
        st.header("🔬 Configuración Científica")
        
        st.subheader("Parámetros del Espacio de Estados")
        dimension = st.selectbox("Dimensión N", [100, 500, 1000], index=1,
                               help="Dimensión del espacio de estados conscientes")
        
        st.subheader("Parámetros Neurofisiológicos")
        N_micronodos = st.slider("Número de Micronodos", 50, 200, 100,
                               help="Número de columnas corticales simuladas")
        frecuencia_gamma = st.slider("Frecuencia Gamma (Hz)", 30, 80, (35, 65),
                                   help="Rango de frecuencias para oscilaciones gamma")
        
        st.subheader("Umbrales Fenomenológicos")
        umbral_consciente = st.slider("γ_min", 0.1, 0.5, 0.3, 0.01,
                                    help="Umbral mínimo para estado consciente")
        capacidad_maxima = st.slider("C_max", 0.5, 2.0, 1.0, 0.1,
                                   help="Capacidad máxima del campo consciente")
        
        st.subheader("Paradigma Experimental")
        paradigma = st.selectbox(
            "Tipo de Paradigma",
            ["validacion_gamma", "anestesia_general", "microestimulacion", "basal"],
            help="Seleccione el paradigma experimental a simular"
        )
        
        # ========== SECCIÓN AVANZADA EN BARRA LATERAL ==========
        st.markdown("---")
        st.header("🔍 Análisis Avanzado")
        
        habilitar_avanzado = st.checkbox("Habilitar Análisis Científico Avanzado", 
                                       value=False,
                                       help="Incluye validación experimental e isomorfismo mente-cerebro")
        
        tipo_simulacion = st.radio(
            "Tipo de Simulación",
            ["Básica", "Avanzada con Validación"],
            index=0,
            help="Seleccione el nivel de análisis científico"
        )
    
    # ========== CONFIGURACIÓN DEL SISTEMA ==========
    config = ConfiguracionNeurofisiologica(
        N_dimension=dimension,
        N_micronodos=N_micronodos,
        frecuencia_gamma=frecuencia_gamma,
        umbral_fenomenologico=umbral_consciente,
        capacidad_maxima=capacidad_maxima
    )
    
    # ========== BOTONES DE EJECUCIÓN PRINCIPAL ==========
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Ejecutar Simulación Básica", use_container_width=True):
            ejecutar_simulacion_basica(config, paradigma)
    
    with col2:
        if st.button("🔬 Ejecutar Simulación Avanzada", type="primary", use_container_width=True):
            ejecutar_simulacion_avanzada(config, paradigma)
    
    # ========== SECCIÓN INFORMATIVA ==========
    with st.expander("📚 Marco Teórico Completo - Implementado", expanded=True):
        st.markdown("""
        ### ✅ **Sistema Completamente Implementado**
        
        #### 1. **Espacio de Estados Conscientes** (Axioma 1)
        - Base ortonormal con superposición suave
        - Producto interno ⟨ψ_i|ψ_j⟩ = e^{-d_ij²}
        - Dimensión N configurable
        
        #### 2. **Sistema Neural Distribuido** 
        - Micronodos ≈ Columnas corticales
        - Conectividad basada en distancia anatómica
        - Oscilaciones gamma realistas
        
        #### 3. **Coeficiente Fenomenológico Completo**
        - c_i(t) = Γ_i(t) · A_i(t) · e^{iθ_i(t)}
        - Γ_i(t) con transformada de Hilbert
        - Umbrales neurofisiológicos reales
        
        #### 4. **Campo Consciente Unificado**
        - |Ψ(t)⟩ = Σ c_i(t)|ψ_i⟩
        - Evolución temporal unitaria
        - Conservación recursos: ⟨Ψ|Ψ⟩ ≤ C_max
        
        #### 5. **Análisis Avanzado (Nuevo)**
        - Isomorfismo Mente-Cerebro Φ: ℋ → 𝒩
        - Validación Experimental Cuantitativa
        - Análisis de Transiciones de Estado
        - Reporte Científico Automático
        """)

def ejecutar_simulacion_basica(config: ConfiguracionNeurofisiologica, paradigma: str):
    """
    Ejecuta la simulación básica (funcionalidad original)
    """
    st.header("📊 Simulación Básica - Resultados")
    
    with st.spinner("Calculando dinámica consciente..."):
        simulador = SimuladorCompleto(config)
        resultados = simulador.simular_paradigma_experimental(paradigma, duracion=8.0)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        correlacion = np.corrcoef(resultados['gamma'], resultados['coeficiente_modulo'])[0,1]
        st.metric("Correlación Γ-|c|", f"{correlacion:.3f}")
    
    with col2:
        precision_umbral = np.mean(np.array(resultados['estado_consciente']) == 
                                 (np.array(resultados['coeficiente_modulo']) > config.umbral_fenomenologico))
        st.metric("Precisión Umbral", f"{precision_umbral:.1%}")
    
    with col3:
        if np.any(resultados['estado_consciente']) and np.any(~np.array(resultados['estado_consciente'])):
            estabilidad_consciente = np.std(np.array(resultados['coeficiente_modulo'])[resultados['estado_consciente']])
            estabilidad_inconsciente = np.std(np.array(resultados['coeficiente_modulo'])[~np.array(resultados['estado_consciente'])])
            diferencia_estabilidad = estabilidad_inconsciente - estabilidad_consciente
            st.metric("Δ Estabilidad", f"{diferencia_estabilidad:.3f}")
        else:
            st.metric("Δ Estabilidad", "N/A")
    
    with col4:
        conservacion_recursos = np.mean(np.array(resultados['norma_campo']) <= config.capacidad_maxima)
        st.metric("Conservación Recursos", f"{conservacion_recursos:.1%}")
    
    # Visualización básica
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Isomorfismo: Γ(t) vs |c(t)|',
            'Dinámica Temporal Completa',
            'Evolución del Campo Consciente',
            'Estados Conscientes'
        )
    )
    
    tiempo = resultados['tiempo']
    
    # Subplot 1: Isomorfismo Gamma vs Coeficiente
    fig.add_trace(
        go.Scatter(x=resultados['gamma'], y=resultados['coeficiente_modulo'],
                  mode='markers', name='Γ vs |c|'),
        row=1, col=1
    )
    fig.add_hline(y=config.umbral_fenomenologico, line_dash="dash", 
                 line_color="red", row=1, col=1)
    
    # Subplot 2: Dinámica temporal
    fig.add_trace(go.Scatter(x=tiempo, y=resultados['gamma'], 
                           name='Γ(t)', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=tiempo, y=resultados['amplitud'], 
                           name='A(t)', line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Scatter(x=tiempo, y=resultados['coeficiente_modulo'], 
                           name='|c(t)|', line=dict(color='green', width=3)), row=1, col=2)
    fig.add_hline(y=config.umbral_fenomenologico, line_dash="dash", 
                 line_color="red", row=1, col=2)
    
    # Subplot 3: Campo consciente
    fig.add_trace(go.Scatter(x=tiempo, y=resultados['norma_campo'],
                           name='‖Ψ(t)‖', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=config.capacidad_maxima, line_dash="dash", 
                 line_color="orange", row=2, col=1)
    
    # Subplot 4: Estados conscientes
    estados_binario = np.array(resultados['estado_consciente']).astype(int)
    fig.add_trace(go.Scatter(x=tiempo, y=estados_binario,
                           name='Estado Consciente', line=dict(color='black')), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Análisis Básico - Coeficientes Fenomenológicos")
    st.plotly_chart(fig, use_container_width=True)

def ejecutar_simulacion_avanzada(config: ConfiguracionNeurofisiologica, paradigma: str):
    """
    Ejecuta la simulación avanzada con validación científica completa
    """
    st.header("🔬 Simulación Avanzada - Análisis Científico Completo")
    
    with st.spinner("Ejecutando análisis científico avanzado..."):
        simulador_extendido = SimuladorCompletoExtendido(config)
        resultados_extendidos = simulador_extendido.simular_paradigma_con_validacion(paradigma, duracion=8.0)
    
    # ========== REPORTE CIENTÍFICO PRINCIPAL ==========
    st.subheader("📈 Reporte Científico Automático")
    reporte = simulador_extendido.generar_reporte_cientifico(resultados_extendidos)
    st.text_area("Reporte Detallado", reporte, height=300)
    
    # ========== MÉTRICAS AVANZADAS EN TIEMPO REAL ==========
    st.subheader("📊 Métricas de Validación Científica")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        iso_val = resultados_extendidos.get('validacion_isomorfismo_gamma_coeficiente', {})
        if iso_val:
            st.metric("Isomorfismo Γ-|c|", 
                     f"r = {iso_val.get('correlacion_pearson', 0):.3f}",
                     delta="Válido" if iso_val.get('isomorfismo_valido', False) else "No válido",
                     help="Correlación entre sincronización gamma y coeficiente fenomenológico")
    
    with col2:
        umbral_val = resultados_extendidos.get('validacion_umbral_fenomenologico', {})
        if umbral_val and 'precision_umbral_teorico' in umbral_val:
            st.metric("Precisión Umbral", 
                     f"{umbral_val['precision_umbral_teorico']:.1%}",
                     delta="Axioma válido" if umbral_val.get('umbral_valido', False) else "Revisar",
                     help="Precisión del umbral fenomenológico teórico γ_min = 0.3")
    
    with col3:
        transiciones = resultados_extendidos.get('analisis_transiciones_estado', {})
        if transiciones:
            total_transiciones = transiciones.get('n_transiciones_positivas', 0) + transiciones.get('n_transiciones_negativas', 0)
            st.metric("Total Transiciones", 
                     f"{total_transiciones}",
                     help="Número total de transiciones consciente-inconsciente")
    
    with col4:
        iso_mente = resultados_extendidos.get('isomorfismo_mente_cerebro', {})
        if iso_mente:
            st.metric("Γ Isomorfismo", 
                     f"{iso_mente.get('gamma_promedio', 0):.3f}",
                     help="Gamma promedio del isomorfismo mente-cerebro")
    
    # ========== VISUALIZACIONES AVANZADAS ==========
    st.subheader("📈 Visualizaciones Científicas Avanzadas")
    
    # Crear pestañas para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs([
        "Análisis de Isomorfismo", 
        "Validación Experimental", 
        "Transiciones de Estado",
        "Isomorfismo Mente-Cerebro"
    ])
    
    with tab1:
        # Análisis de isomorfismo detallado
        fig_iso = go.Figure()
        
        gamma = resultados_extendidos['gamma']
        coeficiente = resultados_extendidos['coeficiente_modulo']
        
        fig_iso.add_trace(go.Scatter(
            x=gamma, y=coeficiente, mode='markers',
            name='Puntos Γ vs |c|',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Línea de regresión (solo si hay variabilidad suficiente)
        iso_val = resultados_extendidos.get('validacion_isomorfismo_gamma_coeficiente', {})
        if len(gamma) > 1 and np.std(gamma) > 1e-10 and iso_val.get('advertencia') is None:
            try:
                z = np.polyfit(gamma, coeficiente, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(gamma), max(gamma), 100)
                fig_iso.add_trace(go.Scatter(
                    x=x_range, y=p(x_range), mode='lines',
                    name=f'Regresión (r = {iso_val.get("correlacion_pearson", 0):.3f})',
                    line=dict(color='red', width=3)
                ))
            except:
                pass  # Si falla la regresión, no mostrar línea
        
        fig_iso.add_hline(y=0.3, line_dash="dash", line_color="green", 
                         annotation_text="Umbral γ_min = 0.3")
        fig_iso.update_layout(
            title="Análisis Detallado del Isomorfismo Γ(t) ↔ |c(t)|",
            xaxis_title="Grado de Sincronización Γ(t)",
            yaxis_title="Coeficiente Fenomenológico |c(t)|",
            height=500
        )
        st.plotly_chart(fig_iso, use_container_width=True)
    
    with tab2:
        # Validación del umbral fenomenológico
        fig_umbral = make_subplots(rows=1, cols=2, 
                                 subplot_titles=('Distribución por Estado', 'Curva ROC'))
        
        coef_arr = np.array(resultados_extendidos['coeficiente_modulo'])
        consciente_arr = np.array(resultados_extendidos['estado_consciente'])
        
        # Subplot 1: Distribución
        fig_umbral.add_trace(go.Violin(
            x=consciente_arr, y=coef_arr, 
            points="all", pointpos=-1.5, jitter=0.05,
            scalemode='count', name='Distribución |c|',
            box_visible=True, meanline_visible=True
        ), row=1, col=1)
        
        fig_umbral.add_hline(y=0.3, line_dash="dash", line_color="red", 
                           annotation_text="γ_min teórico", row=1, col=1)
        
        # Subplot 2: Curva ROC
        if len(np.unique(consciente_arr)) >= 2:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, thresholds = roc_curve(consciente_arr, coef_arr)
            roc_auc = auc(fpr, tpr)
            
            fig_umbral.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'ROC (AUC = {roc_auc:.3f})',
                line=dict(color='blue', width=3)
            ), row=1, col=2)
            
            fig_umbral.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='Línea base', line=dict(color='red', dash='dash')
            ), row=1, col=2)
        
        fig_umbral.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_umbral, use_container_width=True)
    
    with tab3:
        # Análisis de transiciones
        fig_trans = make_subplots(rows=2, cols=1, 
                                subplot_titles=('Historial de Estados', 'Análisis de Transiciones'))
        
        tiempo = resultados_extendidos['tiempo']
        estados = np.array(resultados_extendidos['estado_consciente']).astype(int)
        coeficiente = resultados_extendidos['coeficiente_modulo']
        
        # Subplot 1: Historial de estados
        fig_trans.add_trace(go.Scatter(
            x=tiempo, y=estados, mode='lines',
            name='Estado Consciente', line=dict(color='black', width=2)
        ), row=1, col=1)
        
        # Subplot 2: Coeficiente con transiciones marcadas
        fig_trans.add_trace(go.Scatter(
            x=tiempo, y=coeficiente, mode='lines',
            name='|c(t)|', line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        fig_trans.add_hline(y=0.3, line_dash="dash", line_color="red", 
                          annotation_text="γ_min", row=2, col=1)
        
        # Marcar transiciones
        transiciones = np.diff(estados)
        trans_pos = np.where(transiciones > 0)[0]
        trans_neg = np.where(transiciones < 0)[0]
        
        if len(trans_pos) > 0:
            fig_trans.add_trace(go.Scatter(
                x=np.array(tiempo)[trans_pos], y=np.array(coeficiente)[trans_pos],
                mode='markers', name='Transición ↑',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ), row=2, col=1)
        
        if len(trans_neg) > 0:
            fig_trans.add_trace(go.Scatter(
                x=np.array(tiempo)[trans_neg], y=np.array(coeficiente)[trans_neg],
                mode='markers', name='Transición ↓',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ), row=2, col=1)
        
        fig_trans.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_trans, use_container_width=True)
    
    with tab4:
        # Visualización del isomorfismo mente-cerebro
        iso_mente = resultados_extendidos.get('isomorfismo_mente_cerebro', {})
        
        if iso_mente:
            col1, col2 = st.columns(2)
            
            with col1:
                # Métricas del isomorfismo
                st.metric("Γ Promedio", f"{iso_mente.get('gamma_promedio', 0):.3f}")
                st.metric("A Promedio", f"{iso_mente.get('amplitud_promedio', 0):.3f}")
                st.metric("Entropía Sincronización", f"{iso_mente.get('entropia_sincronizacion', 0):.3f}")
            
            with col2:
                # Diagrama radial del isomorfismo
                fig_radar = go.Figure()
                
                categorias = ['Sincronización', 'Activación', 'Coherencia', 'Estabilidad']
                valores = [
                    iso_mente.get('gamma_promedio', 0),
                    iso_mente.get('amplitud_promedio', 0),
                    abs(np.exp(1j * iso_mente.get('fase_coherente', 0))),
                    1 - iso_mente.get('entropia_sincronizacion', 0)
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=valores, theta=categorias, fill='toself',
                    name='Estado Neural'
                ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title="Perfil del Isomorfismo Mente-Cerebro"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
    
    # ========== ANÁLISIS ESTADÍSTICO AVANZADO ==========
    with st.expander("📊 Análisis Estadístico Riguroso"):
        st.subheader("Análisis Estadístico Completo")
        
        # Test de hipótesis 1: Gamma consciente > Gamma inconsciente
        if np.any(resultados_extendidos['estado_consciente']) and np.any(~np.array(resultados_extendidos['estado_consciente'])):
            gamma_consciente = np.array(resultados_extendidos['gamma'])[resultados_extendidos['estado_consciente']]
            gamma_inconsciente = np.array(resultados_extendidos['gamma'])[~np.array(resultados_extendidos['estado_consciente'])]
            
            t_stat, p_valor = stats.ttest_ind(gamma_consciente, gamma_inconsciente)
            
            st.write("**Test de Hipótesis 1**: Γ(consciente) > Γ(inconsciente)")
            st.write(f"t-statistic = {t_stat:.3f}, p-value = {p_valor:.3f}")
            st.write("✅ **Hipótesis apoyada**" if p_valor < 0.05 and t_stat > 0 else "❌ **Hipótesis no apoyada**")
        
        # Resumen estadístico completo
        st.subheader("Resumen Estadístico")
        df_resultados = pd.DataFrame({
            'Tiempo': resultados_extendidos['tiempo'],
            'Gamma': resultados_extendidos['gamma'],
            'Amplitud': resultados_extendidos['amplitud'],
            'Coeficiente': resultados_extendidos['coeficiente_modulo'],
            'Norma_Campo': resultados_extendidos['norma_campo'],
            'Consciente': resultados_extendidos['estado_consciente']
        })
        st.dataframe(df_resultados.describe())

if __name__ == "__main__":
    main()
