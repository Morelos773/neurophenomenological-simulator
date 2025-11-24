"""
Basic functionality tests for the Neurophenomenological Simulator
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_actual import (
    ConfiguracionNeurofisiologica,
    EspacioEstadosConscientes,
    SistemaMicronodosNeurales,
    CoeficienteFenomenologicoCompleto
)

class TestBasicFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration"""
        self.config = ConfiguracionNeurofisiologica(
            N_dimension=100,
            N_micronodos=50,
            frecuencia_gamma=(30, 80),
            umbral_fenomenologico=0.3
        )
    
    def test_configuration_creation(self):
        """Test that configuration is created correctly"""
        self.assertEqual(self.config.N_dimension, 100)
        self.assertEqual(self.config.N_micronodos, 50)
        self.assertEqual(self.config.frecuencia_gamma, (30, 80))
        self.assertEqual(self.config.umbral_fenomenologico, 0.3)
    
    def test_conscious_space_creation(self):
        """Test conscious state space initialization"""
        espacio = EspacioEstadosConscientes(self.config)
        
        self.assertEqual(espacio.dimension, 100)
        self.assertEqual(espacio.estados_base.shape, (100, 100))
        
        # Test that basis is approximately orthonormal
        for i in range(min(5, espacio.dimension)):
            estado = espacio.estado_qualia_elemental(i)
            norm = espacio.norma(estado)
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_neural_system_initialization(self):
        """Test neural system initialization"""
        sistema = SistemaMicronodosNeurales(self.config)
        
        self.assertEqual(len(sistema.micronodos['fases']), 50)
        self.assertEqual(len(sistema.micronodos['amplitudes']), 50)
        self.assertEqual(sistema.matriz_conectividad.shape, (50, 50))
        
        # Test connectivity matrix properties
        self.assertTrue(np.all(sistema.matriz_conectividad >= 0))
        self.assertTrue(np.all(sistema.matriz_conectividad <= 1))
    
    def test_coefficient_calculation(self):
        """Test phenomenological coefficient calculation"""
        sistema = SistemaMicronodosNeurales(self.config)
        coeficiente = CoeficienteFenomenologicoCompleto(sistema)
        
        # Test with sample data
        fases = np.random.uniform(0, 2*np.pi, 50)
        amplitudes = np.random.uniform(0.1, 0.3, 50)
        tiempo = 1.0
        
        c_complejo = coeficiente.coeficiente_complejo(fases, amplitudes, tiempo)
        
        # Coefficient should be complex and within bounds
        self.assertIsInstance(c_complejo, complex)
        self.assertTrue(0 <= abs(c_complejo) <= 1)
    
    def test_gamma_calculation(self):
        """Test gamma synchronization calculation"""
        sistema = SistemaMicronodosNeurales(self.config)
        coeficiente = CoeficienteFenomenologicoCompleto(sistema)
        
        # Test with perfectly synchronized phases
        fases_sincronizadas = np.ones(50) * np.pi/4
        gamma = coeficiente.calcular_gamma(fases_sincronizadas, 1.0)
        
        self.assertTrue(0 <= gamma <= 1)
        
        # Test with random phases (should have lower gamma)
        fases_aleatorias = np.random.uniform(0, 2*np.pi, 50)
        gamma_random = coeficiente.calcular_gamma(fases_aleatorias, 1.0)
        
        self.assertTrue(0 <= gamma_random <= 1)

def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()