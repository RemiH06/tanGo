"""
weight_engine.py
----------------
Motor de pesos — corazón de la lógica de tanGo.

Diseño: todas las funciones son PURAS.
  - No guardan estado.
  - No llaman APIs.
  - No escriben a base de datos.
  - Dado el mismo input, siempre devuelven el mismo output.

Esto las hace 100% testeables sin mocks y fáciles de razonar.
La lógica de "cuándo cambiar el semáforo" vive aquí, no en Intersection.

Reglas de peso implementadas (a completar):

  Vías (RoadSegment):
    - Fin de semana          → base_weight × 0.7 (distribuir tráfico)
    - Hora pico              → base_weight × 1.2
    - Madrugada (00-05h)     → base_weight × 0.5 (menos tráfico esperado)
    - Lluvia intensa         → base_weight × 0.8 (conducción más lenta)

  Vehículos (Vehicle):
    - Madrugada              → base_weight × 1.5 (más prioridad a autos)
    - EMERGENCY              → retornar base_weight sin modificar
    - Lluvia + BICYCLE       → base_weight × 1.3

  Peatones (Pedestrian):
    - Temperatura < 5°C      → base_weight × 1.3 (mayor vulnerabilidad)
    - Temperatura > 35°C     → base_weight × 1.3
    - Lluvia                 → base_weight × 1.3
    - Madrugada              → base_weight × 0.8
    - is_wheelchair          → siempre base_weight × 1.5 (mínimo)
"""

from __future__ import annotations
from typing import List, Sequence

from core.context import TrafficContext
from core.entities import TrafficEntity, Vehicle, Pedestrian, VehicleType
from core.road import RoadSegment, Intersection


class WeightEngine:
    """
    Motor de pesos. Instanciar una vez y reutilizar — es stateless.
    Todos los métodos son efectivamente funciones puras envueltas
    en una clase para organización y para facilitar la inyección
    de dependencias en los tests.
    """

    # ── Modificadores de vía ─────────────────────────────────────────────────

    def compute_road_weight(self, road: RoadSegment,
                            ctx: TrafficContext) -> float:
        """
        Calcula el peso efectivo de un segmento vial dado el contexto.

        Parameters
        ----------
        road : Segmento a evaluar.
        ctx  : Contexto ambiental del ciclo.

        Returns
        -------
        Peso efectivo de la vía como float positivo.
        """
        # TODO: aplicar modificadores según ctx
        raise NotImplementedError

    # ── Modificadores de entidad ──────────────────────────────────────────────

    def compute_entity_weight(self, entity: TrafficEntity,
                              ctx: TrafficContext) -> float:
        """
        Delega al método compute_weight() de cada entidad
        y aplica modificadores globales adicionales si aplica.

        Parameters
        ----------
        entity : Entidad a evaluar (Vehicle o Pedestrian).
        ctx    : Contexto ambiental del ciclo.

        Returns
        -------
        Peso efectivo de la entidad.
        """
        # TODO: llamar entity.compute_weight(ctx) + modificadores globales
        raise NotImplementedError

    # ── Agregación de presión ─────────────────────────────────────────────────

    def aggregate_pressure(self, entities: Sequence[TrafficEntity],
                           intersection: Intersection,
                           ctx: TrafficContext) -> float:
        """
        Suma los pesos efectivos de todas las entidades presentes
        en una intersección, normalizados contra el peso de la vía.

        Fórmula base:
            presión = Σ compute_entity_weight(e, ctx)
                      / compute_road_weight(segmento_principal, ctx)
                      × 100

        Una presión ≥ 100 indica que hay suficiente demanda para
        justificar un cambio de fase.

        Parameters
        ----------
        entities     : Entidades presentes en la intersección ahora.
        intersection : Intersección a evaluar.
        ctx          : Contexto ambiental del ciclo.

        Returns
        -------
        Presión normalizada (0.0 – ∞). Valor > 100 → cambiar fase.
        """
        # TODO: implementar agregación y normalización
        raise NotImplementedError

    # ── Decisión de fase ──────────────────────────────────────────────────────

    def should_change_phase(self, pressure: float,
                            threshold: float = 100.0) -> bool:
        """
        Decide si la presión acumulada justifica cambiar la fase.
        Función pura más simple del sistema — fácil de testear.

        Parameters
        ----------
        pressure  : Presión calculada por aggregate_pressure().
        threshold : Umbral de cambio (default 100.0).

        Returns
        -------
        True si pressure >= threshold.
        """
        # TODO: implementar — considerar histéresis para evitar oscilación
        raise NotImplementedError

    # ── Ola verde ─────────────────────────────────────────────────────────────

    def compute_green_wave_offset(self, distance_m: float,
                                  speed_limit_kmh: float) -> float:
        """
        Calcula el offset en segundos para que el semáforo vecino
        cambie a verde justo cuando los vehículos lleguen.

        Fórmula: offset = (distance_m / (speed_limit_kmh / 3.6))

        Parameters
        ----------
        distance_m      : Distancia entre las dos intersecciones.
        speed_limit_kmh : Velocidad límite del segmento entre ellas.

        Returns
        -------
        Offset en segundos (float).
        """
        # TODO: implementar
        raise NotImplementedError