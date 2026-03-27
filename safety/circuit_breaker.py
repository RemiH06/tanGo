"""
safety/circuit_breaker.py
--------------------------
Patrón Circuit Breaker para el pipeline de ingesta.

Estados:
  CLOSED  → funcionamiento normal, las llamadas pasan.
  OPEN    → demasiados fallos, las llamadas se bloquean y se usa fallback.
  HALF_OPEN → se intenta una llamada de prueba para ver si el servicio recuperó.

Por qué es crítico en un sistema de semáforos:
  Si TomTom o Open-Meteo fallan, el pipeline NO debe caerse.
  El semáforo debe seguir funcionando con los últimos pesos conocidos
  (fallback) hasta que el servicio externo se recupere.

Ciberseguridad:
  También protege contra ataques de denegación de servicio que intenten
  saturar las APIs externas usando el pipeline como intermediario.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, Optional, TypeVar, Any

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED    = auto()   # Normal — llamadas permitidas
    OPEN      = auto()   # Fallo — llamadas bloqueadas, usando fallback
    HALF_OPEN = auto()   # Prueba — una llamada de prueba permitida


@dataclass
class CircuitBreaker:
    """
    Circuit Breaker genérico para cualquier llamada externa.

    Attributes
    ----------
    name             : Nombre del servicio protegido (para logs).
    failure_threshold: Número de fallos consecutivos antes de abrir.
    recovery_timeout : Segundos antes de intentar HALF_OPEN desde OPEN.
    """

    name:              str
    failure_threshold: int           = 3
    recovery_timeout:  int           = 30   # segundos

    _state:         CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int          = field(default=0, init=False)
    _last_failure:  Optional[datetime] = field(default=None, init=False)

    @property
    def state(self) -> CircuitState:
        """Estado actual del circuit breaker."""
        return self._state

    def call(self, func: Callable[[], T],
             fallback: Callable[[], T]) -> T:
        """
        Ejecuta func() si el circuito está CLOSED o HALF_OPEN.
        Si está OPEN (o func() falla), ejecuta fallback().

        Parameters
        ----------
        func     : Función que llama al servicio externo.
        fallback : Función que retorna datos de respaldo seguros.

        Returns
        -------
        Resultado de func() o fallback() según el estado.
        """
        # TODO: implementar lógica de estados y transiciones
        raise NotImplementedError

    def _on_success(self) -> None:
        """
        Llamar cuando func() tiene éxito.
        Resetea el contador de fallos y cierra el circuito.
        """
        # TODO: implementar
        raise NotImplementedError

    def _on_failure(self) -> None:
        """
        Llamar cuando func() lanza una excepción.
        Incrementa el contador; si supera threshold, abre el circuito.
        """
        # TODO: implementar
        raise NotImplementedError

    def _should_attempt_reset(self) -> bool:
        """
        True si ha pasado suficiente tiempo desde el último fallo
        para intentar una llamada de prueba (HALF_OPEN).
        """
        # TODO: implementar verificación de recovery_timeout
        raise NotImplementedError