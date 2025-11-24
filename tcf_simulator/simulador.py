import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple
from .configuracion import ConfiguracionNeurofisiologica
from .espacio import EspacioEstadosConscientes
from .sistema import SistemaMicronodosNeurales
from .coeficientes import CoeficienteFenomenologicoCompleto
from .campo import CampoConscienteUnificado
from .isomorfismo import IsomorfismoMenteCerebro
from .validador import ValidadorExperimental
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

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
        
        # If streamlit not available or running headless, we avoid progress UI breaking tests
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
        except Exception:
            progress_bar = None
            status_text = None
        
        tiempos = np.arange(0, duracion, self.config.resolucion_temporal)
        total_pasos = len(tiempos)

        for idx, t in enumerate(tiempos):
            if status_text is not None:
                status_text.text(f"Simulando tiempo t = {t:.2f} s")
            if progress_bar is not None:
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

        if progress_bar is not None:
            progress_bar.progress(100)
        if status_text is not None:
            status_text.text("Simulación completada.")

        transiciones = np.diff(resultados['estado_consciente']).astype(int)
        resultados['transiciones_estado'] = transiciones.tolist()
        
        return resultados

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
                # guardamos módulo y fase final del coeficiente
                if len(resultados_original['coeficiente_modulo'])>0 and len(resultados_original['coeficiente_fase'])>0:
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
        if 'tiempo' in resultados and len(resultados['tiempo'])>0:
            reporte.append(f"Duración de simulación: {resultados['tiempo'][-1]:.2f} segundos")
        reporte.append("")
        # agregamos datos clave del validador si existen
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
        umbral = resultados.get('validacion_umbral_fenomenologico', {})
        if umbral and 'precision_umbral_teorico' in umbral:
            reporte.append("2. VALIDACIÓN UMBRAL FENOMENOLÓGICO γ_min ≈ 0.3:")
            reporte.append(f"   - Precisión del umbral teórico: {umbral.get('precision_umbral_teorico',0):.1%}")
        reporte.append("=== FIN DEL REPORTE ===")
        return "\n".join(reporte)
