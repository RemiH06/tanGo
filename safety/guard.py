"""
safety/guard.py
---------------
Capa de validación que se ejecuta ANTES de cualquier cambio de fase.

Responsabilidades:
  1. Rechazar cambios de fase inseguros (tiempo mínimo de verde no cumplido).
  2. Extender el verde automáticamente para peatones vulnerables.
  3. Bloquear giros prohibidos en el grafo.
  4. Manejar overrides de emergencia (ambulancias, bomberos).
  5. Garantizar tiempo mínimo de rojo para que la intersección se despeje.

SafetyGuard es la última línea de defensa antes de emitir una señal.
Ningún cambio de fase debe ocurrir sin pasar por aquí.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from core.context import TrafficContext
from core.entities import TrafficEntity, Pedestrian, Vehicle, VehicleType
from core.road import Intersection, Phase, Turn, RoadSegment


# Tiempo mínimo absoluto de verde en segundos (hardcoded por seguridad física)
MIN_GREEN_SECONDS: int = 7
# Tiempo mínimo de rojo para despejar la intersección
MIN_RED_SECONDS: int   = 5
# Velocidad de cruce para silla de ruedas (m/s)
WHEELCHAIR_SPEED_MS: float = 0.8
# Velocidad de cruce peatón estándar (m/s)
PEDESTRIAN_SPEED_MS: float = 1.2
# Buffer de seguridad adicional al final del cruce (segundos)
CROSSING_BUFFER_S: int = 3


@dataclass
class PhaseChangeRequest:
    """
    Solicitud de cambio de fase que SafetyGuard debe aprobar o rechazar.

    Attributes
    ----------
    intersection   : Intersección que solicita el cambio.
    requested_phase: Fase solicitada por WeightEngine.
    entities       : Entidades presentes en este ciclo.
    ctx            : Contexto ambiental.
    """
    intersection:    Intersection
    requested_phase: Phase
    entities:        List[TrafficEntity]
    ctx:             TrafficContext


@dataclass
class PhaseChangeResult:
    """
    Resultado de la validación de SafetyGuard.

    Attributes
    ----------
    approved        : True si el cambio puede ejecutarse.
    final_phase     : Fase aprobada (puede diferir de la solicitada).
    green_duration_s: Duración de verde en segundos (si aplica).
    reason          : Razón de aprobación o rechazo (para logs de auditoría).
    """
    approved:         bool
    final_phase:      Phase
    green_duration_s: Optional[int]
    reason:           str


class SafetyGuard:
    """
    Valida y aprueba cambios de fase antes de emitirlos.
    Stateless — se puede instanciar una vez y reutilizar.
    """

    def validate(self, request: PhaseChangeRequest) -> PhaseChangeResult:
        """
        Punto de entrada principal. Ejecuta todas las validaciones
        en orden de prioridad:
          1. Emergency override (prioridad máxima, no se puede rechazar)
          2. Tiempo mínimo de verde/rojo
          3. Extensión para peatones vulnerables
          4. Giros prohibidos

        Parameters
        ----------
        request : Solicitud de cambio de fase a validar.

        Returns
        -------
        PhaseChangeResult con la decisión final.
        """
        # TODO: implementar pipeline de validaciones
        raise NotImplementedError

    def check_emergency_override(
            self, entities: List[TrafficEntity]) -> Optional[Phase]:
        """
        Si hay un vehículo de emergencia presente, retorna la fase
        que debe forzarse (normalmente GREEN para el carril de emergencia).
        Retorna None si no hay emergencia.

        Parameters
        ----------
        entities : Entidades presentes en la intersección.

        Returns
        -------
        Phase forzada o None.
        """
        # TODO: implementar detección de VehicleType.EMERGENCY
        raise NotImplementedError

    def enforce_min_green(self, intersection: Intersection,
                          requested_phase: Phase) -> bool:
        """
        Verifica que el semáforo haya estado en verde el tiempo mínimo
        antes de permitir el cambio a amarillo/rojo.

        Parameters
        ----------
        intersection   : Intersección a validar.
        requested_phase: Fase solicitada.

        Returns
        -------
        True si el cambio está permitido (tiempo mínimo cumplido).
        """
        # TODO: implementar verificación de tiempo transcurrido
        raise NotImplementedError

    def compute_wheelchair_extension(
            self, entities: List[TrafficEntity],
            crossing_width_m: float) -> int:
        """
        Si hay peatones en silla de ruedas, calcula los segundos
        adicionales de verde necesarios para que crucen con seguridad.

        Fórmula:
            t = (crossing_width_m / WHEELCHAIR_SPEED_MS) + CROSSING_BUFFER_S

        Parameters
        ----------
        entities         : Entidades en la intersección.
        crossing_width_m : Ancho del cruce en metros.

        Returns
        -------
        Segundos adicionales de verde (0 si no hay sillas de ruedas).
        """
        # TODO: implementar
        raise NotImplementedError

    def is_turn_safe(self, segment: RoadSegment, turn: Turn) -> bool:
        """
        Verifica que un giro sea legal en el segmento dado.
        Caso esquina: vuelta en U, giro en rojo, etc.

        Parameters
        ----------
        segment : Segmento de donde viene el vehículo.
        turn    : Giro que intenta realizar.

        Returns
        -------
        True si el giro está permitido.
        """
        return segment.is_turn_allowed(turn)