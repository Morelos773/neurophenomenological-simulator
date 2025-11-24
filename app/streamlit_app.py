import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

from tcf_simulator.configuracion import ConfiguracionNeurofisiologica
from tcf_simulator.analisis import (
    ejecutar_simulacion_basica,
    ejecutar_simulacion_avanzada
)

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
