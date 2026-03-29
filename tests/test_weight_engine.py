"""
tests/test_weight_engine.py
----------------------------
Tests unitarios para WeightEngine.

Al ser funciones puras, todos los tests son deterministas:
no necesitan mocks de red, base de datos ni tiempo real.

Ejecutar:
    pytest tests/test_weight_engine.py -v
"""

from datetime import datetime
import pytest

from core.context import TrafficContext
from core.entities import Vehicle, Pedestrian, VehicleType, Direction
from core.road import RoadSegment, Intersection, RoadCategory, Phase
from core.weight_engine import WeightEngine


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> WeightEngine:
    return WeightEngine()


@pytest.fixture
def ctx_rush_hour() -> TrafficContext:
    """Contexto: lunes 8am, soleado, temperatura agradable."""
    return TrafficContext(
        timestamp       = datetime(2024, 3, 4, 8, 0),   # lunes 8am
        temperature_c   = 22.0,
        is_raining      = False,
        wind_speed_kmh  = 10.0,
        visibility_m    = 10000.0,
        is_weekend      = False,
        is_rush_hour    = True,
        is_late_night   = False,
    )


@pytest.fixture
def ctx_late_night() -> TrafficContext:
    """Contexto: miércoles 2am, despejado."""
    return TrafficContext(
        timestamp       = datetime(2024, 3, 6, 2, 0),
        temperature_c   = 18.0,
        is_raining      = False,
        wind_speed_kmh  = 5.0,
        visibility_m    = 10000.0,
        is_weekend      = False,
        is_rush_hour    = False,
        is_late_night   = True,
    )


@pytest.fixture
def ctx_rainy_weekend() -> TrafficContext:
    """Contexto: sábado 3pm, lluvia."""
    return TrafficContext(
        timestamp       = datetime(2024, 3, 9, 15, 0),
        temperature_c   = 16.0,
        is_raining      = True,
        wind_speed_kmh  = 25.0,
        visibility_m    = 3000.0,
        is_weekend      = True,
        is_rush_hour    = False,
        is_late_night   = False,
    )


@pytest.fixture
def main_avenue() -> RoadSegment:
    return RoadSegment(
        segment_id      = "seg-av-principal",
        from_node_id    = "A1",
        to_node_id      = "A2",
        category        = RoadCategory.MAIN_AVENUE,
        length_m        = 300.0,
        speed_limit_kmh = 60.0,
    )


@pytest.fixture
def side_street() -> RoadSegment:
    return RoadSegment(
        segment_id      = "seg-calle-lateral",
        from_node_id    = "B1",
        to_node_id      = "B2",
        category        = RoadCategory.STREET,
        length_m        = 100.0,
        speed_limit_kmh = 30.0,
    )


# ── Tests: pesos de vía ───────────────────────────────────────────────────────

class TestRoadWeight:

    def test_main_avenue_base_weight(self, engine, main_avenue, ctx_rush_hour):
        """La avenida principal tiene peso mayor que una calle secundaria."""
        side = RoadSegment("s", "X", "Y", RoadCategory.STREET, 100, 30)
        w_avenue = engine.compute_road_weight(main_avenue, ctx_rush_hour)
        w_street = engine.compute_road_weight(side, ctx_rush_hour)
        assert w_avenue > w_street

    def test_weekend_reduces_avenue_weight(self, engine, main_avenue,
                                            ctx_rush_hour, ctx_rainy_weekend):
        """En fin de semana la avenida principal tiene menos peso."""
        w_weekday = engine.compute_road_weight(main_avenue, ctx_rush_hour)
        w_weekend = engine.compute_road_weight(main_avenue, ctx_rainy_weekend)
        assert w_weekend < w_weekday

    def test_late_night_reduces_road_weight(self, engine, main_avenue,
                                             ctx_rush_hour, ctx_late_night):
        """De madrugada el peso de la vía baja (menos tráfico esperado)."""
        w_rush = engine.compute_road_weight(main_avenue, ctx_rush_hour)
        w_night = engine.compute_road_weight(main_avenue, ctx_late_night)
        assert w_night < w_rush


# ── Tests: pesos de entidades ─────────────────────────────────────────────────

class TestEntityWeight:

    def test_pedestrian_heavier_than_car(self, engine, ctx_rush_hour):
        """Un peatón individual pesa más que un auto individual."""
        car = Vehicle("v1", VehicleType.CAR, Direction.NORTH)
        ped = Pedestrian("p1")
        w_car = engine.compute_entity_weight(car, ctx_rush_hour)
        w_ped = engine.compute_entity_weight(ped, ctx_rush_hour)
        assert w_ped > w_car

    def test_late_night_boosts_vehicle_weight(self, engine,
                                               ctx_rush_hour, ctx_late_night):
        """De madrugada los vehículos tienen más prioridad."""
        car = Vehicle("v1", VehicleType.CAR, Direction.NORTH)
        w_day   = engine.compute_entity_weight(car, ctx_rush_hour)
        w_night = engine.compute_entity_weight(car, ctx_late_night)
        assert w_night > w_day

    def test_rain_boosts_pedestrian_weight(self, engine,
                                            ctx_rush_hour, ctx_rainy_weekend):
        """Lluvia aumenta el peso del peatón."""
        ped = Pedestrian("p1")
        w_dry   = engine.compute_entity_weight(ped, ctx_rush_hour)
        w_rainy = engine.compute_entity_weight(ped, ctx_rainy_weekend)
        assert w_rainy > w_dry

    def test_wheelchair_always_has_highest_pedestrian_weight(
            self, engine, ctx_late_night):
        """Silla de ruedas tiene el mayor peso incluso de madrugada."""
        ped_normal = Pedestrian("p1", is_wheelchair=False)
        ped_wheel  = Pedestrian("p2", is_wheelchair=True)
        w_normal = engine.compute_entity_weight(ped_normal, ctx_late_night)
        w_wheel  = engine.compute_entity_weight(ped_wheel, ctx_late_night)
        assert w_wheel > w_normal


# ── Tests: presión de intersección ────────────────────────────────────────────

class TestPressure:

    def test_nine_pedestrians_vs_main_avenue(self, engine, ctx_rush_hour):
        """
        Caso del enunciado: avenida w=90, necesita 9 peatones (w=10 c/u)
        para alcanzar presión >= 100.
        """
        # Intersección sobre la avenida principal
        avenue_seg = RoadSegment(
            "seg-av", "A1", "A2", RoadCategory.MAIN_AVENUE, 300, 60
        )
        intersection = Intersection(
            node_id  = "A1",
            name     = "Av. Principal y Calle 5",
            latitude = 20.67,
            longitude= -103.35,
            incoming_segments = [avenue_seg],
        )
        # 8 peatones — NO debe superar el umbral
        eight_peds = [Pedestrian(f"p{i}") for i in range(8)]
        p8 = engine.aggregate_pressure(eight_peds, intersection, ctx_rush_hour)
        assert not engine.should_change_phase(p8), \
            "8 peatones no deberían superar el umbral en avenida principal"

        # 9 peatones — DEBE superar el umbral
        nine_peds = [Pedestrian(f"p{i}") for i in range(9)]
        p9 = engine.aggregate_pressure(nine_peds, intersection, ctx_rush_hour)
        assert engine.should_change_phase(p9), \
            "9 peatones deberían superar el umbral en avenida principal"

    def test_fewer_pedestrians_needed_in_rain(self, engine,
                                               ctx_rush_hour, ctx_rainy_weekend):
        """
        Con lluvia, los peatones pesan más — se necesitan menos
        para alcanzar el umbral en la misma avenida.
        """
        avenue_seg = RoadSegment(
            "seg-av", "A1", "A2", RoadCategory.MAIN_AVENUE, 300, 60
        )
        intersection = Intersection(
            "A1", "Test", 20.67, -103.35, [avenue_seg]
        )
        peds = [Pedestrian(f"p{i}") for i in range(7)]

        p_dry   = engine.aggregate_pressure(peds, intersection, ctx_rush_hour)
        p_rainy = engine.aggregate_pressure(peds, intersection, ctx_rainy_weekend)

        assert p_rainy > p_dry, \
            "La misma cantidad de peatones debe generar más presión con lluvia"