"""tcf_simulator package - modularized from main_actual.py"""
from .configuracion import ConfiguracionNeurofisiologica
from .espacio import EspacioEstadosConscientes
from .sistema import SistemaMicronodosNeurales
from .coeficientes import CoeficienteFenomenologicoCompleto
from .campo import CampoConscienteUnificado
from .simulador import SimuladorCompleto, SimuladorCompletoExtendido
from .isomorfismo import IsomorfismoMenteCerebro
from .validador import ValidadorExperimental

__all__ = [
    "ConfiguracionNeurofisiologica",
    "EspacioEstadosConscientes",
    "SistemaMicronodosNeurales",
    "CoeficienteFenomenologicoCompleto",
    "CampoConscienteUnificado",
    "SimuladorCompleto",
    "SimuladorCompletoExtendido",
    "IsomorfismoMenteCerebro",
    "ValidadorExperimental",
]
