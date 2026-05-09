"""
dags/tango_queries_dag.py
--------------------------
Pipeline de actualización de datos tanGo — KAN-10.

Flujo HORARIO (cada hora):
  inicio
    → verificar_grafo     ← comprueba city_graph.json (si >24h → descarga)
    → refrescar_grafo     ← descarga desde Overpass solo si es necesario
    → enriquecer_contexto ← TomTom (velocidades) + Open-Meteo (clima)
    → correr_simulacion   ← 10-15 ticks con TrafficAlgorithm + contexto real
    → exportar_estado     ← escribe tango_state.json para FastAPI
                            (solo nodos semaforizados — sin blind)
  fin

NOTA: calcular_pesos se separó a tango_daily_dag.py (DAG diario).
Los pesos estáticos (betweenness, pagerank) no cambian cada hora —
calcularlos cada run era costoso e innecesario.

Fuentes externas:
  - Overpass API    — grafo OSM (sin key)
  - Open-Meteo API  — clima en tiempo real (sin key, gratis)
  - TomTom API      — velocidades reales por segmento (key en .env)

VisionIngester (KAN-16) se conectará reemplazando el spawn
sintético en correr_simulacion cuando esté disponible.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

# ── Path al proyecto ──────────────────────────────────────────────────────────
TANGO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TANGO_ROOT))

_env_file = TANGO_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

CITY_JSON    = TANGO_ROOT / "graph" / "city_graph.json"
STATE_JSON   = TANGO_ROOT / "graph" / "tango_state.json"
CONTEXT_JSON = TANGO_ROOT / "graph" / "tango_context.json"

# FIX: número de ticks por corrida — suficiente para que los semáforos
# cambien de fase y el estado refleje algo real, sin ser demasiado lento.
N_TICKS_PER_RUN = 15

logger = logging.getLogger(__name__)

default_args = {
    "owner":           "Rem",
    "depends_on_past": False,
    "retries":         1,
    "retry_delay":     timedelta(minutes=3),
}


# ── Tarea 1: Verificar grafo ──────────────────────────────────────────────────

def verificar_grafo(**context) -> dict:
    if not CITY_JSON.exists():
        return {"needs_refresh": True, "reason": "no_existe"}
    mtime = datetime.fromtimestamp(CITY_JSON.stat().st_mtime)
    age_h = (datetime.now() - mtime).total_seconds() / 3600
    logger.info("city_graph.json edad: %.1f horas", age_h)
    if age_h > 24:
        return {"needs_refresh": True, "reason": f"desactualizado_{age_h:.0f}h"}
    return {"needs_refresh": False, "age_h": age_h}


# ── Tarea 2: Refrescar grafo ──────────────────────────────────────────────────

def refrescar_grafo(**context) -> dict:
    ti     = context["ti"]
    status = ti.xcom_pull(task_ids="verificar_grafo")

    if status and status.get("needs_refresh"):
        logger.info("Descargando grafo desde Overpass...")
        from graph.city_loader import download_graph, process_graph, radius_to_bbox

        lat = float(os.environ.get("CITY_LATITUDE",  "20.6597"))
        lon = float(os.environ.get("CITY_LONGITUDE", "-103.3496"))
        radius_m = float(os.environ.get("CITY_RADIUS_M", "800"))

        bbox = radius_to_bbox(lat, lon, radius_m)
        logger.info("Bbox: %s", bbox)

        raw  = download_graph(bbox)
        data = process_graph(raw)

        with open(CITY_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        n_nodes = data["metadata"]["n_nodes"]
        logger.info("Grafo descargado: %d nodos", n_nodes)
    else:
        with open(CITY_JSON, encoding="utf-8") as f:
            data = json.load(f)
        n_nodes = data["metadata"]["n_nodes"]
        logger.info("Grafo existente: %d nodos", n_nodes)

    return {"n_nodes": n_nodes}


# ── Tarea 3: Enriquecer contexto (TomTom + Open-Meteo) ───────────────────────

def enriquecer_contexto(**context) -> dict:
    """
    Obtiene datos reales de:
      - Open-Meteo: temperatura, lluvia, viento, visibilidad
      - TomTom: velocidades actuales por segmento del grafo

    Guarda tango_context.json para que correr_simulacion lo use.
    Si alguna API falla, usa valores por defecto — el pipeline no se detiene.
    """
    import requests

    lat        = float(os.environ.get("CITY_LATITUDE",  "20.6597"))
    lon        = float(os.environ.get("CITY_LONGITUDE", "-103.3496"))
    tomtom_key = os.environ.get("TOMTOM_API_KEY", "")

    # ── Open-Meteo ────────────────────────────────────────────────────────────
    weather = {
        "temperature_c":  22.0,
        "is_raining":     False,
        "wind_speed_kmh": 10.0,
        "visibility_m":   10000.0,
        "source":         "default",
    }
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,precipitation,wind_speed_10m,visibility"
            f"&wind_speed_unit=kmh&timezone=auto"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        curr = resp.json().get("current", {})
        weather = {
            "temperature_c":  curr.get("temperature_2m",   22.0),
            "is_raining":     curr.get("precipitation",     0.0) > 0.1,
            "wind_speed_kmh": curr.get("wind_speed_10m",   10.0),
            "visibility_m":   curr.get("visibility",     10000.0),
            "source":         "open-meteo",
        }
        logger.info(
            "Open-Meteo: %.1f°C lluvia=%s viento=%.1fkm/h",
            weather["temperature_c"], weather["is_raining"],
            weather["wind_speed_kmh"],
        )
    except Exception as e:
        logger.warning("Open-Meteo falló, usando defaults: %s", e)

    # ── TomTom Traffic ────────────────────────────────────────────────────────
    traffic_factor = 1.0
    road_speeds: dict[str, float] = {}

    if tomtom_key and tomtom_key != "your_32_char_key_here":
        try:
            with open(CITY_JSON, encoding="utf-8") as f:
                data = json.load(f)

            edges     = data.get("edges", [])[:20]
            nodes_map = {n["node_id"]: n for n in data.get("nodes", [])}

            total_ratio = 0.0
            count = 0

            for edge in edges:
                from_n = nodes_map.get(edge["from_node_id"])
                to_n   = nodes_map.get(edge["to_node_id"])
                if not from_n or not to_n:
                    continue

                mid_lat = (from_n["latitude"]  + to_n["latitude"])  / 2
                mid_lon = (from_n["longitude"] + to_n["longitude"]) / 2

                url = (
                    f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
                    f"absolute/10/json"
                    f"?point={mid_lat},{mid_lon}&key={tomtom_key}"
                )
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    fd            = resp.json().get("flowSegmentData", {})
                    current_speed = fd.get("currentSpeed",  0)
                    free_flow     = fd.get("freeFlowSpeed", 0)
                    if free_flow > 0:
                        ratio = current_speed / free_flow
                        total_ratio += ratio
                        count += 1
                        edge_key = f"{edge['from_node_id']}-{edge['to_node_id']}"
                        road_speeds[edge_key] = current_speed

            if count > 0:
                traffic_factor = round(total_ratio / count, 3)
                logger.info(
                    "TomTom: %d segmentos | factor=%.2f", count, traffic_factor
                )
            else:
                logger.warning("TomTom: sin segmentos válidos")

        except Exception as e:
            logger.warning("TomTom falló, usando factor=1.0: %s", e)
    else:
        logger.info("TomTom: sin API key configurada — omitiendo")

    ctx_data = {
        "timestamp":      datetime.now().isoformat(),
        "latitude":       lat,
        "longitude":      lon,
        "weather":        weather,
        "traffic_factor": traffic_factor,
        "road_speeds":    road_speeds,
        "n_road_speeds":  len(road_speeds),
    }

    with open(CONTEXT_JSON, "w", encoding="utf-8") as f:
        json.dump(ctx_data, f, ensure_ascii=False, indent=2)

    logger.info(
        "Contexto exportado: clima=%s tráfico=%.2f %d velocidades",
        weather["source"], traffic_factor, len(road_speeds),
    )
    return {
        "weather_source":  weather["source"],
        "traffic_factor":  traffic_factor,
        "n_road_speeds":   len(road_speeds),
        "is_raining":      weather["is_raining"],
        "temperature_c":   weather["temperature_c"],
    }


# ── Tarea 4: Correr simulación ────────────────────────────────────────────────

def correr_simulacion(**context) -> dict:
    """
    FIX: corre N_TICKS_PER_RUN ticks (15) en vez de 1.

    Con 1 tick todos los semáforos aparecen en verde porque el algoritmo
    no ha tenido tiempo de acumular presión y cambiar fases. Con 15 ticks
    el estado exportado refleja un ciclo de semáforos real.

    El último tick es el que se exporta a tango_state.json.
    Las métricas (green_count, red_count, etc.) son el promedio del episodio.
    """
    import random
    import uuid

    from core.context   import TrafficContext
    from core.algorithm import TrafficAlgorithm
    from core.entities  import Vehicle, Pedestrian, VehicleType, Direction
    from core.road      import IntersectionType
    from graph.city_loader import json_to_traffic_graph

    # Cargar contexto enriquecido
    ctx_data = {}
    if CONTEXT_JSON.exists():
        with open(CONTEXT_JSON, encoding="utf-8") as f:
            ctx_data = json.load(f)

    weather        = ctx_data.get("weather", {})
    traffic_factor = ctx_data.get("traffic_factor", 1.0)

    ctx = TrafficContext.build(
        timestamp      = datetime.now(),
        temperature_c  = weather.get("temperature_c",  22.0),
        is_raining     = weather.get("is_raining",     False),
        wind_speed_kmh = weather.get("wind_speed_kmh", 10.0),
        visibility_m   = weather.get("visibility_m",   10000.0),
    )

    graph = json_to_traffic_graph(CITY_JSON)
    algo  = TrafficAlgorithm(graph)
    algo.reset()

    congestion_mult = max(0.5, 2.0 - traffic_factor)

    def spawn_node(itype: IntersectionType) -> list:
        if itype == IntersectionType.MASTER:
            nv = int(random.randint(5, 14) * congestion_mult)
        elif itype == IntersectionType.NORMAL:
            nv = int(random.randint(2, 8) * congestion_mult)
        else:
            nv = random.randint(0, 3)

        pool = (
            [VehicleType.CAR]        * 60 +
            [VehicleType.MOTORCYCLE] * 15 +
            [VehicleType.BUS]        * 10 +
            [VehicleType.TRUCK]      * 8  +
            [VehicleType.BICYCLE]    * (2 if ctx.is_raining else 5) +
            [VehicleType.EMERGENCY]  * 2
        )
        ents = [
            Vehicle(str(uuid.uuid4()), random.choice(pool),
                    random.choice(list(Direction)))
            for _ in range(nv)
        ]
        if random.random() < 0.15:
            ents.append(Pedestrian(str(uuid.uuid4())))
        return ents

    # FIX: correr N_TICKS_PER_RUN ticks, acumulando métricas
    metrics_acc = {
        "green_count":  0,
        "red_count":    0,
        "yellow_count": 0,
        "blink_count":  0,
        "total_entities": 0,
    }
    last_result = None

    logger.info("Corriendo %d ticks de simulación...", N_TICKS_PER_RUN)

    for tick_n in range(N_TICKS_PER_RUN):
        entities_by_node = {
            nid: spawn_node(inter.intersection_type)
            for nid, inter in graph.intersections.items()
        }
        result = algo.run_tick(entities_by_node, ctx)
        last_result = result

        metrics_acc["green_count"]    += result.green_count
        metrics_acc["red_count"]      += result.red_count
        metrics_acc["yellow_count"]   += result.yellow_count
        metrics_acc["blink_count"]    += result.blink_count
        metrics_acc["total_entities"] += result.total_entities

    # Promediar métricas del episodio
    metrics_avg = {k: round(v / N_TICKS_PER_RUN, 1)
                   for k, v in metrics_acc.items()}

    logger.info(
        "Simulación completa: %d ticks | %.1f entidades/tick | "
        "verde=%.1f rojo=%.1f factor_tráfico=%.2f",
        N_TICKS_PER_RUN,
        metrics_avg["total_entities"],
        metrics_avg["green_count"],
        metrics_avg["red_count"],
        traffic_factor,
    )

    # Snapshot del ÚLTIMO tick — es el estado más reciente
    intersections_snap = []
    for nid, ns in last_result.nodes.items():
        inter = graph.intersections[nid]
        intersections_snap.append({
            "node_id":     nid,
            "name":        inter.name,
            "latitude":    inter.latitude,
            "longitude":   inter.longitude,
            "phase":       ns.phase,
            "pressure":    round(ns.pressure, 2),
            "node_weight": round(inter.node_weight, 3),
            "has_light":   ns.has_light,
            "itype":       inter.intersection_type.value,
            "neighbors":   list(graph.graph.successors(nid)),
            "counts":      ns.entity_counts,
        })

    return {
        "tick":             last_result.tick_number,
        "n_ticks_run":      N_TICKS_PER_RUN,
        "total_entities":   metrics_avg["total_entities"],
        "green_count":      metrics_avg["green_count"],
        "red_count":        metrics_avg["red_count"],
        "yellow_count":     metrics_avg["yellow_count"],
        "blink_count":      metrics_avg["blink_count"],
        "intersections":    intersections_snap,
        "traffic_factor":   traffic_factor,
        "is_raining":       weather.get("is_raining", False),
    }


# ── Tarea 5: Exportar estado ──────────────────────────────────────────────────

def exportar_estado(**context) -> None:
    """
    FIX: filtra nodos blind del JSON exportado.

    Los nodos blind no tienen semáforo — no aportan nada útil al dashboard
    y generaban confusión al mostrar entidades en intersecciones sin control.
    FastAPI y el dashboard solo ven nodos con has_light=True.
    """
    ti = context["ti"]

    sim_result = ti.xcom_pull(task_ids="correr_simulacion") or {}
    graph_info = ti.xcom_pull(task_ids="refrescar_grafo")   or {}
    ctx_info   = ti.xcom_pull(task_ids="enriquecer_contexto") or {}

    all_intersections = sim_result.get("intersections", [])

    # FIX: solo exportar nodos semaforizados (has_light=True)
    # Los blind se excluyen — no tienen fase que mostrar ni semáforo que controlar
    signaled_intersections = [
        i for i in all_intersections if i.get("has_light")
    ]

    n_blind = len(all_intersections) - len(signaled_intersections)
    logger.info(
        "Filtrando nodos: %d total → %d semaforizados (%d blind excluidos)",
        len(all_intersections), len(signaled_intersections), n_blind,
    )

    state = {
        "updated_at":  datetime.now().isoformat(),
        "n_nodes":     graph_info.get("n_nodes", len(all_intersections)),
        "n_signaled":  len(signaled_intersections),
        "n_blind":     n_blind,
        "n_ticks_run": sim_result.get("n_ticks_run", 1),
        "context": {
            "weather_source":  ctx_info.get("weather_source",  "default"),
            "temperature_c":   ctx_info.get("temperature_c",   22.0),
            "is_raining":      ctx_info.get("is_raining",      False),
            "traffic_factor":  ctx_info.get("traffic_factor",  1.0),
            "n_road_speeds":   ctx_info.get("n_road_speeds",   0),
        },
        "metrics": {
            "total_intersections": len(signaled_intersections),
            "total_records":       sim_result.get("total_entities", 0),
            "n_ticks_run":         sim_result.get("n_ticks_run",    1),
            "green_count":         sim_result.get("green_count",    0),
            "red_count":           sim_result.get("red_count",      0),
            "blink_count":         sim_result.get("blink_count",    0),
            "traffic_factor":      sim_result.get("traffic_factor", 1.0),
        },
        # Solo semaforizados — dashboard y FastAPI consumen esto
        "intersections": signaled_intersections,
    }

    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    logger.info(
        "Estado exportado → %s | %d semaforizados | %d ticks | clima=%s | tráfico=%.2f",
        STATE_JSON,
        len(signaled_intersections),
        sim_result.get("n_ticks_run", 1),
        state["context"]["weather_source"],
        state["context"]["traffic_factor"],
    )


# ── DAG HORARIO ───────────────────────────────────────────────────────────────

with DAG(
    dag_id            = "tango_traffic_pipeline",
    default_args      = default_args,
    description       = "tanGo — pipeline horario: Overpass + TomTom + Open-Meteo + sim",
    start_date        = datetime(2026, 4, 26),
    schedule_interval = "@hourly",
    catchup           = False,
    tags              = ["tanGo", "trafico", "KAN-10"],
) as dag:

    inicio = EmptyOperator(task_id="inicio")
    fin    = EmptyOperator(task_id="fin")

    t_verificar = PythonOperator(task_id="verificar_grafo",     python_callable=verificar_grafo)
    t_refrescar = PythonOperator(task_id="refrescar_grafo",     python_callable=refrescar_grafo)
    t_contexto  = PythonOperator(task_id="enriquecer_contexto", python_callable=enriquecer_contexto)
    t_sim       = PythonOperator(task_id="correr_simulacion",   python_callable=correr_simulacion)
    t_exportar  = PythonOperator(task_id="exportar_estado",     python_callable=exportar_estado)

    inicio >> t_verificar >> t_refrescar >> t_contexto >> t_sim >> t_exportar >> fin