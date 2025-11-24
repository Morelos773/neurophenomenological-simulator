import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

from tcf_simulator.simulador import (
    SimuladorCompleto,
    SimuladorCompletoExtendido
)
from tcf_simulator.configuracion import ConfiguracionNeurofisiologica


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
                           name='Γ(t)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=tiempo, y=resultados['amplitud'], 
                           name='A(t)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=tiempo, y=resultados['coeficiente_modulo'], 
                           name='|c(t)|'), row=1, col=2)
    fig.add_hline(y=config.umbral_fenomenologico, line_dash="dash", 
                 line_color="red", row=1, col=2)
    
    # Subplot 3: Campo consciente
    fig.add_trace(go.Scatter(x=tiempo, y=resultados['norma_campo'],
                           name='‖Ψ(t)‖'), row=2, col=1)
    fig.add_hline(y=config.capacidad_maxima, line_dash="dash", 
                 line_color="orange", row=2, col=1)
    
    # Subplot 4: Estados conscientes
    estados_binario = np.array(resultados['estado_consciente']).astype(int)
    fig.add_trace(go.Scatter(x=tiempo, y=estados_binario,
                           name='Estado Consciente'), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def ejecutar_simulacion_avanzada(config: ConfiguracionNeurofisiologica, paradigma: str):
    """
    Ejecuta la simulación avanzada con validación científica completa
    """
    st.header("🔬 Simulación Avanzada - Análisis Científico Completo")
    
    with st.spinner("Ejecutando análisis científico avanzado..."):
        simulador_extendido = SimuladorCompletoExtendido(config)
        resultados_extendidos = simulador_extendido.simular_paradigma_con_validacion(paradigma, duracion=8.0)
    
    st.subheader("📈 Reporte Científico Automático")
    reporte = simulador_extendido.generar_reporte_cientifico(resultados_extendidos)
    st.text_area("Reporte Detallado", reporte, height=300)
    
    # Métricas avanzadas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        iso_val = resultados_extendidos.get('validacion_isomorfismo_gamma_coeficiente', {})
        if iso_val:
            st.metric("Isomorfismo Γ-|c|", 
                     f"r = {iso_val.get('correlacion_pearson', 0):.3f}")
    
    with col2:
        umbral_val = resultados_extendidos.get('validacion_umbral_fenomenologico', {})
        if umbral_val and 'precision_umbral_teorico' in umbral_val:
            st.metric("Precisión Umbral", 
                     f"{umbral_val['precision_umbral_teorico']:.1%}")
    
    with col3:
        transiciones = resultados_extendidos.get('analisis_transiciones_estado', {})
        if transiciones:
            total_transiciones = transiciones.get('n_transiciones_positivas', 0) + transiciones.get('n_transiciones_negativas', 0)
            st.metric("Total Transiciones", f"{total_transiciones}")
    
    with col4:
        iso_mente = resultados_extendidos.get('isomorfismo_mente_cerebro', {})
        if iso_mente:
            st.metric("Γ Isomorfismo", f"{iso_mente.get('gamma_promedio', 0):.3f}")
