"""
safety/circuit_breaker.py
--------------------------
Patrón Circuit Breaker para el pipeline de ingesta.

Estados:
  CLOSED    → funcionamiento normal, las llamadas pasan.
  OPEN      → demasiados fallos, las llamadas se bloquean y se usa fallback.
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
from typing import Callable, Optional, TypeVar
import logging

logger = logging.getLogger(__name__)

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
    failure_threshold: int               = 3
    recovery_timeout:  int               = 30

    _state:         CircuitState         = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int                  = field(default=0, init=False)
    _last_failure:  Optional[datetime]   = field(default=None, init=False)

    @property
    def state(self) -> CircuitState:
        """Estado actual del circuit breaker."""
        return self._state

    def call(self, func: Callable[[], T], fallback: Callable[[], T]) -> T:
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
        # Si está OPEN, verificar si ya es hora de intentar recuperación
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info("[%s] Circuito HALF_OPEN — intentando recuperación", self.name)
                self._state = CircuitState.HALF_OPEN
            else:
                logger.warning("[%s] Circuito OPEN — usando fallback", self.name)
                return fallback()

        # CLOSED o HALF_OPEN: intentar llamada real
        try:
            result = func()
            self._on_success()
            return result
        except Exception as exc:
            logger.error("[%s] Fallo en llamada externa: %s", self.name, exc)
            self._on_failure()
            return fallback()

    def _on_success(self) -> None:
        """
        Resetea el contador de fallos y cierra el circuito.
        Llamado automáticamente por call() cuando func() tiene éxito.
        """
        if self._state != CircuitState.CLOSED:
            logger.info("[%s] Circuito cerrado — servicio recuperado", self.name)
        self._state         = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure  = None

    def _on_failure(self) -> None:
        """
        Incrementa el contador de fallos.
        Si supera failure_threshold, abre el circuito.
        Llamado automáticamente por call() cuando func() lanza excepción.
        """
        self._failure_count += 1
        self._last_failure   = datetime.now()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                "[%s] Circuito ABIERTO tras %d fallos consecutivos",
                self.name, self._failure_count
            )

    def _should_attempt_reset(self) -> bool:
        """
        True si ha pasado suficiente tiempo desde el último fallo
        para intentar una llamada de prueba (transición a HALF_OPEN).
        """
        if self._last_failure is None:
            return True
        elapsed = datetime.now() - self._last_failure
        return elapsed >= timedelta(seconds=self.recovery_timeout)