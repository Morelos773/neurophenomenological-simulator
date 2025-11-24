"""
Quick Demo - Neurophenomenological Simulator v2.0
Demonstrates basic usage of the advanced simulator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_actual import ConfiguracionNeurofisiologica, SimuladorCompletoExtendido

def run_quick_demo():
    """Run a quick demonstration of the simulator"""
    print("🧠 Neurophenomenological Simulator v2.0 - Quick Demo")
    print("=" * 50)
    
    # Basic configuration
    config = ConfiguracionNeurofisiologica(
        N_dimension=500,
        N_micronodos=80,
        frecuencia_gamma=(35, 65),
        umbral_fenomenologico=0.3,
        capacidad_maxima=1.0
    )
    
    # Create extended simulator
    simulador = SimuladorCompletoExtendido(config)
    
    print("Running gamma validation paradigm...")
    
    # Run simulation with validation
    resultados = simulador.simular_paradigma_con_validacion(
        "validacion_gamma", 
        duracion=5.0  # Shorter for demo
    )
    
    # Generate scientific report
    reporte = simulador.generar_reporte_cientifico(resultados)
    print("\n" + reporte)
    
    # Display key metrics
    print("\n📊 Key Metrics:")
    print(f"- Correlation Γ-|c|: {resultados['validacion_isomorfismo_gamma_coeficiente'].get('correlacion_pearson', 0):.3f}")
    print(f"- Threshold accuracy: {resultados['validacion_umbral_fenomenologico'].get('precision_umbral_teorico', 0):.1%}")
    print(f"- Conscious transitions: {resultados['analisis_transiciones_estado'].get('n_transiciones_positivas', 0)}")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    run_quick_demo()