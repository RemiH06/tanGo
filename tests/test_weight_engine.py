"""
tests/test_weight_engine.py
----------------------------
Tests del WeightEngine y las entidades.

Al ser funciones puras, todos los tests son deterministas —
no requieren mocks, red ni base de datos.

Ejecutar:
    pytest tests/test_weight_engine.py -v
"""

import pytest
from datetime import datetime

from core.context import TrafficContext
from core.entities import Vehicle, Pedestrian, VehicleType, Direction
from core.road import RoadSegment, Intersection, RoadCategory
from core.weight_engine import WeightEngine, _HYSTERESIS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> WeightEngine:
    return WeightEngine()

@pytest.fixture
def ctx_rush() -> TrafficContext:
    return TrafficContext.build(
        timestamp=datetime(2024, 3, 4, 8, 0),
        temperature_c=22.0, is_raining=False,
        wind_speed_kmh=10.0, visibility_m=10000.0,
    )

@pytest.fixture
def ctx_night() -> TrafficContext:
    return TrafficContext.build(
        timestamp=datetime(2024, 3, 4, 2, 0),
        temperature_c=18.0, is_raining=False,
        wind_speed_kmh=5.0, visibility_m=10000.0,
    )

@pytest.fixture
def ctx_rain() -> TrafficContext:
    return TrafficContext.build(
        timestamp=datetime(2024, 3, 4, 14, 0),
        temperature_c=16.0, is_raining=True,
        wind_speed_kmh=20.0, visibility_m=3000.0,
    )

@pytest.fixture
def ctx_weekend() -> TrafficContext:
    return TrafficContext.build(
        timestamp=datetime(2024, 3, 9, 15, 0),
        temperature_c=22.0, is_raining=False,
        wind_speed_kmh=8.0, visibility_m=10000.0,
    )

@pytest.fixture
def ctx_cold() -> TrafficContext:
    return TrafficContext.build(
        timestamp=datetime(2024, 1, 15, 14, 0),
        temperature_c=2.0, is_raining=False,
        wind_speed_kmh=10.0, visibility_m=10000.0,
    )

@pytest.fixture
def main_avenue() -> RoadSegment:
    return RoadSegment(
        "seg-av", "A1", "A2",
        RoadCategory.MAIN_AVENUE, 300.0, 60.0
    )

@pytest.fixture
def side_street() -> RoadSegment:
    return RoadSegment(
        "seg-calle", "B1", "B2",
        RoadCategory.STREET, 100.0, 30.0
    )

@pytest.fixture
def intersection(main_avenue) -> Intersection:
    return Intersection(
        "A1", "Av. Vallarta y López Mateos",
        20.6597, -103.3496,
        incoming_segments=[main_avenue],
    )


# ── Tests: peso de vía ────────────────────────────────────────────────────────

class TestRoadWeight:

    def test_avenue_heavier_than_street(self, engine, main_avenue,
                                        side_street, ctx_rush):
        w_av = engine.compute_road_weight(main_avenue, ctx_rush)
        w_st = engine.compute_road_weight(side_street, ctx_rush)
        assert w_av > w_st

    def test_rush_hour_increases_weight(self, engine, main_avenue,
                                        ctx_rush, ctx_night):
        w_rush  = engine.compute_road_weight(main_avenue, ctx_rush)
        w_night = engine.compute_road_weight(main_avenue, ctx_night)
        assert w_rush > w_night

    def test_weekend_reduces_weight(self, engine, main_avenue,
                                    ctx_rush, ctx_weekend):
        w_weekday = engine.compute_road_weight(main_avenue, ctx_rush)
        w_weekend = engine.compute_road_weight(main_avenue, ctx_weekend)
        assert w_weekend < w_weekday

    def test_late_night_reduces_weight(self, engine, main_avenue,
                                       ctx_rush, ctx_night):
        w_rush  = engine.compute_road_weight(main_avenue, ctx_rush)
        w_night = engine.compute_road_weight(main_avenue, ctx_night)
        assert w_night < w_rush

    def test_rain_reduces_road_weight(self, engine, main_avenue,
                                      ctx_rush, ctx_rain):
        w_dry   = engine.compute_road_weight(main_avenue, ctx_rush)
        w_rainy = engine.compute_road_weight(main_avenue, ctx_rain)
        assert w_rainy < w_dry

    def test_weight_never_zero(self, engine, main_avenue, ctx_night):
        assert engine.compute_road_weight(main_avenue, ctx_night) > 0


# ── Tests: peso de entidades ──────────────────────────────────────────────────

class TestEntityWeight:

    def test_pedestrian_heavier_than_car(self, engine, ctx_rush):
        car = Vehicle("v1", VehicleType.CAR, Direction.NORTH)
        ped = Pedestrian("p1")
        assert engine.compute_entity_weight(ped, ctx_rush) > \
               engine.compute_entity_weight(car, ctx_rush)

    def test_vehicle_heavier_at_night(self, engine, ctx_rush, ctx_night):
        car = Vehicle("v1", VehicleType.CAR, Direction.NORTH)
        w_day   = engine.compute_entity_weight(car, ctx_rush)
        w_night = engine.compute_entity_weight(car, ctx_night)
        assert w_night > w_day

    def test_pedestrian_heavier_in_rain(self, engine, ctx_rush, ctx_rain):
        ped = Pedestrian("p1")
        w_dry   = engine.compute_entity_weight(ped, ctx_rush)
        w_rainy = engine.compute_entity_weight(ped, ctx_rain)
        assert w_rainy > w_dry

    def test_pedestrian_lighter_at_night(self, engine, ctx_rush, ctx_night):
        ped = Pedestrian("p1")
        w_day   = engine.compute_entity_weight(ped, ctx_rush)
        w_night = engine.compute_entity_weight(ped, ctx_night)
        assert w_night < w_day

    def test_wheelchair_always_heavier(self, engine, ctx_night):
        """Silla de ruedas tiene mayor peso incluso de madrugada."""
        ped_normal = Pedestrian("p1", is_wheelchair=False)
        ped_wheel  = Pedestrian("p2", is_wheelchair=True)
        assert engine.compute_entity_weight(ped_wheel,  ctx_night) > \
               engine.compute_entity_weight(ped_normal, ctx_night)

    def test_cold_increases_pedestrian_weight(self, engine, ctx_rush, ctx_cold):
        ped = Pedestrian("p1")
        w_warm = engine.compute_entity_weight(ped, ctx_rush)
        w_cold = engine.compute_entity_weight(ped, ctx_cold)
        assert w_cold > w_warm

    def test_bicycle_heavier_in_rain(self, engine, ctx_rush, ctx_rain):
        bike = Vehicle("v1", VehicleType.BICYCLE, Direction.NORTH)
        w_dry   = engine.compute_entity_weight(bike, ctx_rush)
        w_rainy = engine.compute_entity_weight(bike, ctx_rain)
        assert w_rainy > w_dry

    def test_emergency_weight_unchanged_by_context(self, engine,
                                                    ctx_rush, ctx_night):
        """El peso de emergencia nunca se modifica — siempre 999."""
        emerg = Vehicle("v1", VehicleType.EMERGENCY, Direction.NORTH)
        assert engine.compute_entity_weight(emerg, ctx_rush)  == 999.0
        assert engine.compute_entity_weight(emerg, ctx_night) == 999.0


# ── Tests: required_green_seconds ────────────────────────────────────────────

class TestRequiredGreenSeconds:

    def test_wheelchair_needs_more_time(self):
        ped_normal = Pedestrian("p1", crossing_width_m=10.0)
        ped_wheel  = Pedestrian("p2", is_wheelchair=True, crossing_width_m=10.0)
        assert ped_wheel.required_green_seconds() > ped_normal.required_green_seconds()

    def test_wider_crossing_needs_more_time(self):
        ped_narrow = Pedestrian("p1", crossing_width_m=5.0)
        ped_wide   = Pedestrian("p2", crossing_width_m=15.0)
        assert ped_wide.required_green_seconds() > ped_narrow.required_green_seconds()

    def test_normal_pedestrian_10m(self):
        # 10m / 1.2 m/s + 3s buffer = 11.33s
        ped = Pedestrian("p1", crossing_width_m=10.0)
        assert ped.required_green_seconds() == pytest.approx(11.33, rel=0.01)

    def test_wheelchair_10m(self):
        # 10m / 0.8 m/s + 3s buffer = 15.5s
        ped = Pedestrian("p1", is_wheelchair=True, crossing_width_m=10.0)
        assert ped.required_green_seconds() == pytest.approx(15.5, rel=0.01)


# ── Tests: aggregate_pressure — casos del enunciado ──────────────────────────

class TestAggregatePressure:

    def test_no_entities_returns_zero(self, engine, intersection, ctx_rush):
        assert engine.aggregate_pressure([], intersection, ctx_rush) == 0.0

    def test_emergency_returns_max_pressure(self, engine, intersection, ctx_rush):
        emerg = Vehicle("v1", VehicleType.EMERGENCY, Direction.NORTH)
        pressure = engine.aggregate_pressure([emerg], intersection, ctx_rush)
        assert pressure == 999.0

    def test_nine_pedestrians_trigger_change_on_main_avenue(
            self, engine):
        """
        Caso del enunciado:
        Avenida principal (w base=80), 9 peatones (w base=10 c/u).

        Con pesos base sin modificadores de contexto:
            8 peds: (80 / 80) × 100 = 100.0 → exactamente en el umbral
            9 peds: (90 / 80) × 100 = 112.5 → supera el umbral

        El test verifica que:
          1. 9 peatones sí generan un cambio de fase.
          2. 9 peatones generan más presión que 8.
          3. La presión escala linealmente con el número de entidades.

        Nota: en contexto con modificadores (hora pico, lluvia, etc.)
        los números exactos varían — el enunciado usó pesos base como referencia.
        """
        ctx_neutral = TrafficContext.build(
            timestamp=datetime(2024, 3, 4, 14, 0),
            temperature_c=22.0, is_raining=False,
            wind_speed_kmh=10.0, visibility_m=10000.0,
        )
        avenue = RoadSegment(
            "seg-av", "A1", "A2",
            RoadCategory.MAIN_AVENUE, 300.0, 60.0
        )
        inter = Intersection(
            "A1", "Test", 20.6597, -103.3496,
            incoming_segments=[avenue],
        )

        eight_peds = [Pedestrian(f"p{i}") for i in range(8)]
        nine_peds  = [Pedestrian(f"p{i}") for i in range(9)]

        p8 = engine.aggregate_pressure(eight_peds, inter, ctx_neutral)
        p9 = engine.aggregate_pressure(nine_peds,  inter, ctx_neutral)

        # 9 peatones SIEMPRE deben superar el umbral con pesos base
        assert engine.should_change_phase(p9), \
            f"9 peatones deben cambiar la fase (presión={p9:.1f})"

        # La presión debe escalar linealmente
        assert p9 > p8, \
            f"9 peatones deben tener más presión que 8 ({p9:.1f} > {p8:.1f})"

        # Verificar valores exactos con pesos base
        assert p8 == pytest.approx(100.0), f"8 peds → presión esperada 100.0, got {p8:.2f}"
        assert p9 == pytest.approx(112.5), f"9 peds → presión esperada 112.5, got {p9:.2f}"

    def test_rain_fewer_pedestrians_needed(self, engine, ctx_rush, ctx_rain):
        """Con lluvia los peatones pesan más — se necesitan menos para cambiar."""
        avenue = RoadSegment(
            "seg-av", "A1", "A2",
            RoadCategory.MAIN_AVENUE, 300.0, 60.0
        )
        inter = Intersection(
            "A1", "Test", 20.6597, -103.3496,
            incoming_segments=[avenue],
        )
        peds = [Pedestrian(f"p{i}") for i in range(7)]
        p_dry   = engine.aggregate_pressure(peds, inter, ctx_rush)
        p_rainy = engine.aggregate_pressure(peds, inter, ctx_rain)
        assert p_rainy > p_dry

    def test_pressure_higher_at_night_for_vehicles(self, engine,
                                                    ctx_rush, ctx_night):
        """De noche los vehículos pesan más → mayor presión."""
        avenue = RoadSegment(
            "seg-av", "A1", "A2",
            RoadCategory.MAIN_AVENUE, 300.0, 60.0
        )
        inter = Intersection(
            "A1", "Test", 20.6597, -103.3496,
            incoming_segments=[avenue],
        )
        cars = [Vehicle(f"v{i}", VehicleType.CAR, Direction.NORTH)
                for i in range(5)]
        p_day   = engine.aggregate_pressure(cars, inter, ctx_rush)
        p_night = engine.aggregate_pressure(cars, inter, ctx_night)
        assert p_night > p_day


# ── Tests: should_change_phase ────────────────────────────────────────────────

class TestShouldChangePhase:

    def test_above_threshold(self, engine):
        assert engine.should_change_phase(100.0) is True

    def test_below_threshold(self, engine):
        assert engine.should_change_phase(99.9) is False

    def test_exactly_threshold(self, engine):
        assert engine.should_change_phase(100.0) is True

    def test_custom_threshold(self, engine):
        assert engine.should_change_phase(50.0, threshold=50.0) is True
        assert engine.should_change_phase(49.9, threshold=50.0) is False

    def test_emergency_always_changes(self, engine):
        assert engine.should_change_phase(999.0) is True


# ── Tests: green wave offset ──────────────────────────────────────────────────

class TestGreenWaveOffset:

    def test_basic_calculation(self, engine):
        # 300m / (60 km/h → 16.67 m/s) ≈ 18s
        offset = engine.compute_green_wave_offset(300.0, 60.0)
        assert offset == pytest.approx(18.0, rel=0.01)

    def test_shorter_distance_smaller_offset(self, engine):
        assert engine.compute_green_wave_offset(100.0, 60.0) < \
               engine.compute_green_wave_offset(300.0, 60.0)

    def test_faster_speed_smaller_offset(self, engine):
        assert engine.compute_green_wave_offset(300.0, 80.0) < \
               engine.compute_green_wave_offset(300.0, 60.0)

    def test_zero_speed_raises(self, engine):
        with pytest.raises(ValueError):
            engine.compute_green_wave_offset(300.0, 0.0)


# ── Tests: congestion factor ──────────────────────────────────────────────────

class TestCongestionFactor:

    def test_free_flow_factor_is_one(self, engine):
        assert engine.congestion_to_pressure_factor(0.0) == pytest.approx(1.0)

    def test_full_congestion_factor(self, engine):
        assert engine.congestion_to_pressure_factor(1.0) == pytest.approx(2.5)

    def test_factor_increases_with_congestion(self, engine):
        assert engine.congestion_to_pressure_factor(0.3) < \
               engine.congestion_to_pressure_factor(0.7)

    def test_out_of_range_clamped(self, engine):
        assert engine.congestion_to_pressure_factor(-0.5) == pytest.approx(1.0)
        assert engine.congestion_to_pressure_factor(1.5)  == pytest.approx(2.5)