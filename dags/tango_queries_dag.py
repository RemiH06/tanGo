"""
dags/tango_queries_dag.py
--------------------------
Pipeline de actualización de datos tanGo — KAN-10.

Flujo (cada hora):
  inicio
    → verificar_grafo     ← comprueba city_graph.json
    → refrescar_grafo     ← descarga desde Overpass si es necesario
    → enriquecer_contexto ← TomTom (velocidades) + Open-Meteo (clima)
    → calcular_pesos      ← betweenness, pagerank, road_quality
    → correr_simulacion   ← 1 tick con TrafficAlgorithm + contexto real
    → exportar_estado     ← escribe tango_state.json para FastAPI
  fin

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

# Cargar .env si existe
_env_file = TANGO_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

CITY_JSON  = TANGO_ROOT / "graph" / "city_graph.json"
STATE_JSON = TANGO_ROOT / "graph" / "tango_state.json"
CONTEXT_JSON = TANGO_ROOT / "graph" / "tango_context.json"

logger = logging.getLogger(__name__)

default_args = {
    "owner":           "diego",
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
        from graph.city_loader import process_graph, load_config
        cfg  = load_config()
        data = process_graph(cfg)
        with open(CITY_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        n_nodes = len(data.get("nodes", []))
        logger.info("Grafo descargado: %d nodos", n_nodes)
    else:
        with open(CITY_JSON, encoding="utf-8") as f:
            data = json.load(f)
        n_nodes = len(data.get("nodes", []))
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

    lat = float(os.environ.get("CITY_LATITUDE",  "20.6597"))
    lon = float(os.environ.get("CITY_LONGITUDE", "-103.3496"))
    tomtom_key = os.environ.get("TOMTOM_API_KEY", "")

    # ── Open-Meteo (sin key, gratis) ──────────────────────────────────────────
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
            weather["wind_speed_kmh"]
        )
    except Exception as e:
        logger.warning("Open-Meteo falló, usando defaults: %s", e)

    # ── TomTom Traffic (con key) ──────────────────────────────────────────────
    traffic_factor = 1.0   # 1.0 = velocidad libre, <1.0 = congestión
    road_speeds: dict[str, float] = {}

    if tomtom_key and tomtom_key != "your_32_char_key_here":
        try:
            with open(CITY_JSON, encoding="utf-8") as f:
                data = json.load(f)

            # Tomar una muestra de segmentos para no exceder el rate limit
            edges = data.get("edges", [])[:20]
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
                    f"?point={mid_lat},{mid_lon}"
                    f"&key={tomtom_key}"
                )
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    fd = resp.json().get("flowSegmentData", {})
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
                    "TomTom: %d segmentos | factor de flujo=%.2f",
                    count, traffic_factor
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
        weather["source"], traffic_factor, len(road_speeds)
    )
    return {
        "weather_source":  weather["source"],
        "traffic_factor":  traffic_factor,
        "n_road_speeds":   len(road_speeds),
        "is_raining":      weather["is_raining"],
        "temperature_c":   weather["temperature_c"],
    }


# ── Tarea 4: Calcular pesos ───────────────────────────────────────────────────

def calcular_pesos(**context) -> dict:
    from graph.city_loader import compute_static_weights

    with open(CITY_JSON, encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    logger.info("Calculando pesos estáticos para %d nodos...", len(nodes))
    nodes = compute_static_weights(nodes, edges)
    data["nodes"] = nodes

    with open(CITY_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    weight_stats = {
        "min": round(min(n.get("node_weight", 1.0) for n in nodes), 3),
        "max": round(max(n.get("node_weight", 1.0) for n in nodes), 3),
        "avg": round(sum(n.get("node_weight", 1.0) for n in nodes) / len(nodes), 3),
    }
    logger.info("Pesos calculados: %s", weight_stats)
    return weight_stats


# ── Tarea 5: Correr simulación ────────────────────────────────────────────────

def correr_simulacion(**context) -> dict:
    """
    Corre 1 tick de TrafficAlgorithm con:
      - Contexto real de Open-Meteo (temperatura, lluvia, viento)
      - Factor de tráfico de TomTom (ajusta velocidades de entidades)
      - Spawn sintético (reemplazable por VisionIngester en KAN-16)
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

    # Spawn sintético ajustado por traffic_factor de TomTom
    # traffic_factor < 1 → más tráfico (hora pico real)
    # traffic_factor ≥ 1 → flujo libre
    congestion_mult = max(0.5, 2.0 - traffic_factor)  # inverso suavizado

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

    entities_by_node = {
        nid: spawn_node(inter.intersection_type)
        for nid, inter in graph.intersections.items()
    }

    result = algo.run_tick(entities_by_node, ctx)
    logger.info(
        "Simulación tick %d: %d entidades | %d verdes | factor_tráfico=%.2f",
        result.tick_number, result.total_entities,
        result.green_count, traffic_factor,
    )

    intersections_snap = []
    for nid, ns in result.nodes.items():
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
        "tick":           result.tick_number,
        "total_entities": result.total_entities,
        "green_count":    result.green_count,
        "red_count":      result.red_count,
        "yellow_count":   result.yellow_count,
        "blink_count":    result.blink_count,
        "intersections":  intersections_snap,
        "traffic_factor": traffic_factor,
        "is_raining":     weather.get("is_raining", False),
    }


# ── Tarea 6: Exportar estado ──────────────────────────────────────────────────

def exportar_estado(**context) -> None:
    ti = context["ti"]

    peso_stats = ti.xcom_pull(task_ids="calcular_pesos")    or {}
    sim_result = ti.xcom_pull(task_ids="correr_simulacion") or {}
    graph_info = ti.xcom_pull(task_ids="refrescar_grafo")   or {}
    ctx_info   = ti.xcom_pull(task_ids="enriquecer_contexto") or {}

    intersections = sim_result.get("intersections", [])
    n_signaled    = sum(1 for i in intersections if i.get("has_light"))

    state = {
        "updated_at":     datetime.now().isoformat(),
        "n_nodes":        graph_info.get("n_nodes", len(intersections)),
        "n_signaled":     n_signaled,
        "weight_stats":   peso_stats,
        "context": {
            "weather_source":  ctx_info.get("weather_source",  "default"),
            "temperature_c":   ctx_info.get("temperature_c",   22.0),
            "is_raining":      ctx_info.get("is_raining",      False),
            "traffic_factor":  ctx_info.get("traffic_factor",  1.0),
            "n_road_speeds":   ctx_info.get("n_road_speeds",   0),
        },
        "metrics": {
            "total_intersections": len(intersections),
            "total_records":       sim_result.get("total_entities", 0),
            "total_ticks":         sim_result.get("tick",           0),
            "green_count":         sim_result.get("green_count",    0),
            "red_count":           sim_result.get("red_count",      0),
            "blink_count":         sim_result.get("blink_count",    0),
            "traffic_factor":      sim_result.get("traffic_factor", 1.0),
        },
        "intersections": intersections,
    }

    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    logger.info(
        "Estado exportado → %s | %d intersecciones | clima=%s | tráfico=%.2f",
        STATE_JSON, len(intersections),
        state["context"]["weather_source"],
        state["context"]["traffic_factor"],
    )


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id            = "tango_traffic_graph_pipeline",
    default_args      = default_args,
    description       = "Pipeline tanGo — Overpass + TomTom + Open-Meteo + simulación",
    start_date        = datetime(2026, 4, 26),
    schedule_interval = "@hourly",
    catchup           = False,
    tags              = ["tanGo", "grafos", "trafico", "KAN-10"],
) as dag:

    inicio = EmptyOperator(task_id="inicio")
    fin    = EmptyOperator(task_id="fin")

    t_verificar  = PythonOperator(task_id="verificar_grafo",     python_callable=verificar_grafo)
    t_refrescar  = PythonOperator(task_id="refrescar_grafo",     python_callable=refrescar_grafo)
    t_contexto   = PythonOperator(task_id="enriquecer_contexto", python_callable=enriquecer_contexto)
    t_pesos      = PythonOperator(task_id="calcular_pesos",      python_callable=calcular_pesos)
    t_sim        = PythonOperator(task_id="correr_simulacion",   python_callable=correr_simulacion)
    t_exportar   = PythonOperator(task_id="exportar_estado",     python_callable=exportar_estado)

    inicio >> t_verificar >> t_refrescar >> t_contexto >> t_pesos >> t_sim >> t_exportar >> fin