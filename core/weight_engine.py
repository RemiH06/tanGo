"""
weight_engine.py
----------------
Motor de pesos — corazón de la lógica de tanGo.

Diseño: todas las funciones son PURAS.
  - No guardan estado.
  - No llaman APIs ni escriben a base de datos.
  - Dado el mismo input → siempre el mismo output.
  - Testeables sin mocks.

Flujo por ciclo:
  1. compute_road_weight(segmento, ctx)      → peso efectivo de la vía
  2. compute_entity_weight(entidad, ctx)     → peso efectivo de cada entidad
  3. aggregate_pressure(entidades, inter, ctx) → presión total normalizada
  4. should_change_phase(presión)            → bool → cambiar o no
"""

from __future__ import annotations
from typing import Sequence
import logging

from core.context import TrafficContext
from core.entities import TrafficEntity, Vehicle, Pedestrian, VehicleType
from core.road import RoadSegment, Intersection

logger = logging.getLogger(__name__)

# Umbral de histéresis — evita que el semáforo oscile si la presión
# ronda exactamente el valor de threshold.
# Para cambiar a verde se necesita presión >= threshold + HYSTERESIS
# Para mantenerse verde se necesita presión >= threshold - HYSTERESIS
_HYSTERESIS: float = 5.0

# Peso mínimo de vía para evitar división por cero
_MIN_ROAD_WEIGHT: float = 1.0


class WeightEngine:
    """
    Motor de pesos — stateless, instanciar una vez y reutilizar.

    Todos los métodos reciben todo lo que necesitan como parámetros
    y no modifican ningún estado externo. Esto los hace funciones
    puras en la práctica aunque estén envueltos en una clase.
    """

    # ── Peso de vía ───────────────────────────────────────────────────────────

    def compute_road_weight(self, road: RoadSegment,
                            ctx: TrafficContext) -> float:
        """
        Calcula el peso efectivo de un segmento vial dado el contexto.

        Modificadores:
          - Hora pico    → × 1.2  (más tráfico, mayor importancia)
          - Fin de semana → × 0.7 (distribuir flujo entre más rutas)
          - Madrugada    → × 0.5  (menos tráfico esperado)
          - Lluvia       → × 0.8  (velocidades más bajas)

        Los modificadores son mutuamente excluyentes en tiempo
        (no puede ser hora pico y madrugada al mismo tiempo).

        Parameters
        ----------
        road : Segmento a evaluar.
        ctx  : Contexto ambiental del ciclo.

        Returns
        -------
        Peso efectivo de la vía — siempre >= _MIN_ROAD_WEIGHT.
        """
        weight = road.base_weight

        # Modificadores temporales (mutuamente excluyentes)
        if ctx.is_late_night:
            weight *= 0.5
        elif ctx.is_rush_hour:
            weight *= 1.2

        # Modificadores independientes
        if ctx.is_weekend:
            weight *= 0.7
        if ctx.is_raining:
            weight *= 0.8

        return max(weight, _MIN_ROAD_WEIGHT)

    # ── Peso de entidad ───────────────────────────────────────────────────────

    def compute_entity_weight(self, entity: TrafficEntity,
                              ctx: TrafficContext) -> float:
        """
        Calcula el peso efectivo de una entidad delegando a su propio
        método compute_weight() e identificando emergencias.

        Si es un vehículo de emergencia retorna el peso directamente
        sin modificadores adicionales — SafetyGuard lo detecta por
        el valor 999 y hace override inmediato.

        Parameters
        ----------
        entity : Entidad a evaluar.
        ctx    : Contexto ambiental del ciclo.

        Returns
        -------
        Peso efectivo de la entidad.
        """
        return entity.compute_weight(ctx)

    # ── Agregación de presión ─────────────────────────────────────────────────

    def aggregate_pressure(self, entities: Sequence[TrafficEntity],
                           intersection: Intersection,
                           ctx: TrafficContext) -> float:
        """
        Calcula la presión total sobre una intersección.

        Fórmula:
            presión = (Σ peso_entidad(e, ctx) / peso_vía(segmento_principal, ctx)) × 100

        Una presión ≥ 100 significa que la demanda de las entidades
        iguala o supera la capacidad/importancia de la vía principal.

        Si no hay entidades → presión = 0.
        Si no hay segmento principal → se usa _MIN_ROAD_WEIGHT como divisor.

        Parameters
        ----------
        entities     : Entidades presentes en la intersección.
        intersection : Intersección a evaluar.
        ctx          : Contexto ambiental del ciclo.

        Returns
        -------
        Presión normalizada en [0, ∞). ≥ 100 → cambiar fase.
        """
        if not entities:
            return 0.0

        # Detectar emergencia — presión máxima inmediata
        for entity in entities:
            if (isinstance(entity, Vehicle)
                    and entity.vehicle_type == VehicleType.EMERGENCY):
                logger.warning(
                    "[%s] Vehículo de emergencia detectado — presión máxima",
                    intersection.name
                )
                return 999.0

        # Suma de pesos de entidades
        total_entity_weight = sum(
            self.compute_entity_weight(e, ctx) for e in entities
        )

        # Peso del segmento principal (el de mayor categoría)
        main_seg = intersection.main_segment()
        road_weight = (
            self.compute_road_weight(main_seg, ctx)
            if main_seg else _MIN_ROAD_WEIGHT
        )

        # weight_multiplier: tipo × geometría (escala el denominador)
        # node_weight: peso estático combinado (centralidad + degree + road_quality)
        #   node_weight=1.0 → intersección promedio de la red
        #   node_weight=1.4 → nodo muy central — necesita más demanda para cambiar
        #   node_weight=0.6 → nodo periférico — cambia con menos presión
        # degree_weight se mantiene por compatibilidad; node_weight lo engloba en sim2+
        road_weight *= intersection.weight_multiplier * intersection.node_weight

        pressure = (total_entity_weight / road_weight) * 100.0

        logger.debug(
            "[%s] presión=%.1f | entidades=%d | Σpeso=%.1f | peso_vía=%.1f",
            intersection.name, pressure,
            len(entities), total_entity_weight, road_weight
        )

        return pressure

    # ── Decisión de fase ──────────────────────────────────────────────────────

    def should_change_phase(self, pressure: float,
                            threshold: float = 100.0) -> bool:
        """
        Decide si la presión acumulada justifica cambiar la fase.

        Usa histéresis para evitar oscilación cuando la presión
        ronda el umbral:
          - Para activar el cambio: pressure >= threshold + HYSTERESIS
          - Esta función evalúa solo si supera el umbral base.
            La histéresis completa se implementa en Intersection._next_phase()

        Parameters
        ----------
        pressure  : Presión calculada por aggregate_pressure().
        threshold : Umbral de cambio (default 100.0).

        Returns
        -------
        True si pressure >= threshold.
        """
        return pressure >= threshold

    # ── Ola verde ─────────────────────────────────────────────────────────────

    def compute_green_wave_offset(self, distance_m: float,
                                  speed_limit_kmh: float) -> float:
        """
        Calcula el offset en segundos para la ola verde.

        El semáforo vecino debe cambiar a verde exactamente cuando
        los vehículos lleguen desde la intersección actual.

        Fórmula:
            offset = distance_m / (speed_limit_kmh / 3.6)

        Parameters
        ----------
        distance_m      : Distancia entre intersecciones en metros.
        speed_limit_kmh : Velocidad límite del segmento.

        Returns
        -------
        Offset en segundos.

        Raises
        ------
        ValueError si speed_limit_kmh <= 0.
        """
        if speed_limit_kmh <= 0:
            raise ValueError(
                f"speed_limit_kmh debe ser positivo, recibido: {speed_limit_kmh}"
            )
        speed_ms = speed_limit_kmh / 3.6
        return distance_m / speed_ms

    # ── Análisis de congestión TomTom ─────────────────────────────────────────

    def congestion_to_pressure_factor(self, congestion_index: float) -> float:
        """
        Convierte el índice de congestión de TomTom (0.0–1.0)
        a un factor multiplicador de presión.

        Mapeo:
          0.0 → 1.0  (vía libre — peso normal)
          0.5 → 1.5  (congestión moderada — aumentar presión)
          1.0 → 2.5  (vía congestionada — máxima presión extra)

        Fórmula lineal: factor = 1.0 + (congestion_index * 1.5)

        Parameters
        ----------
        congestion_index : Índice de TomTom entre 0.0 y 1.0.

        Returns
        -------
        Factor multiplicador ≥ 1.0.
        """
        congestion_index = max(0.0, min(1.0, congestion_index))
        return 1.0 + (congestion_index * 1.5)