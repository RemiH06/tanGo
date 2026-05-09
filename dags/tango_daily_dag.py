"""
dags/tango_daily_dag.py
------------------------
Pipeline DIARIO de tanGo — KAN-10 (complemento).

Responsabilidad: recalcular los pesos estáticos del grafo.

Por qué separado del DAG horario:
  - betweenness, pagerank y road_quality son propiedades de la
    estructura del grafo vial — no cambian hora a hora.
  - Calcularlos en cada run horario era costoso (~30s para 354 nodos)
    e innecesario. Un recálculo diario es suficiente.
  - Si se descarga un grafo nuevo desde Overpass (>24h de antigüedad),
    este DAG recalcula los pesos automáticamente al día siguiente.

Flujo (una vez al día, 3am):
  inicio
    → calcular_pesos  ← betweenness + pagerank + road_quality
    → verificar_pesos ← log de estadísticas para monitoreo
  fin

Dependencia:
  Requiere que city_graph.json exista (generado por el DAG horario
  la primera vez que verificar_grafo detecta que no existe).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

# ── Path al proyecto ──────────────────────────────────────────────────────────
TANGO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TANGO_ROOT))

CITY_JSON = TANGO_ROOT / "graph" / "city_graph.json"

logger = logging.getLogger(__name__)

default_args = {
    "owner":           "Rem",
    "depends_on_past": False,
    "retries":         1,
    "retry_delay":     timedelta(minutes=10),
}


# ── Tarea 1: Calcular pesos estáticos ─────────────────────────────────────────

def calcular_pesos(**context) -> dict:
    """
    Recalcula betweenness, pagerank y road_quality para todos los nodos.

    Usa NetworkX — para 354 nodos tarda ~30 segundos en CPU.
    El resultado se escribe directamente en city_graph.json.

    Returns
    -------
    Dict con estadísticas de los pesos calculados.
    """
    if not CITY_JSON.exists():
        logger.warning(
            "city_graph.json no existe — el DAG horario debe correr primero. "
            "Saltando cálculo de pesos."
        )
        return {"skipped": True, "reason": "city_graph.json no encontrado"}

    from graph.city_loader import compute_static_weights

    with open(CITY_JSON, encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not nodes:
        logger.warning("city_graph.json está vacío o malformado")
        return {"skipped": True, "reason": "sin nodos"}

    logger.info(
        "Calculando pesos estáticos para %d nodos, %d aristas...",
        len(nodes), len(edges)
    )

    nodes = compute_static_weights(nodes, edges)
    data["nodes"] = nodes

    # Actualizar metadata con fecha de último recálculo de pesos
    data.setdefault("metadata", {})["pesos_calculados_at"] = datetime.now().isoformat()

    with open(CITY_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    weight_vals = [n.get("node_weight", 1.0) for n in nodes]
    stats = {
        "n_nodes":   len(nodes),
        "min_weight": round(min(weight_vals), 4),
        "max_weight": round(max(weight_vals), 4),
        "avg_weight": round(sum(weight_vals) / len(weight_vals), 4),
        "skipped":   False,
    }

    logger.info(
        "Pesos calculados: %d nodos | min=%.4f max=%.4f avg=%.4f",
        stats["n_nodes"], stats["min_weight"],
        stats["max_weight"], stats["avg_weight"],
    )
    return stats


# ── Tarea 2: Verificar pesos ──────────────────────────────────────────────────

def verificar_pesos(**context) -> None:
    """
    Log de verificación post-cálculo.
    Imprime los 5 nodos con mayor y menor peso para monitoreo.
    """
    ti     = context["ti"]
    stats  = ti.xcom_pull(task_ids="calcular_pesos") or {}

    if stats.get("skipped"):
        logger.warning("Verificación omitida: %s", stats.get("reason", "desconocido"))
        return

    if not CITY_JSON.exists():
        return

    with open(CITY_JSON, encoding="utf-8") as f:
        data = json.load(f)

    nodes = sorted(
        data.get("nodes", []),
        key=lambda n: n.get("node_weight", 0),
        reverse=True,
    )

    logger.info("── Top 5 nodos por peso ──────────────────────────")
    for n in nodes[:5]:
        logger.info(
            "  %.4f | %s | %s",
            n.get("node_weight", 0),
            n.get("node_id", "?"),
            n.get("name", "Sin nombre")[:40],
        )

    logger.info("── Bottom 5 nodos por peso ───────────────────────")
    for n in nodes[-5:]:
        logger.info(
            "  %.4f | %s | %s",
            n.get("node_weight", 0),
            n.get("node_id", "?"),
            n.get("name", "Sin nombre")[:40],
        )

    logger.info(
        "Verificación completa: %d nodos | rango [%.4f, %.4f]",
        stats.get("n_nodes", 0),
        stats.get("min_weight", 0),
        stats.get("max_weight", 0),
    )


# ── DAG DIARIO ────────────────────────────────────────────────────────────────

with DAG(
    dag_id            = "tango_daily_weights",
    default_args      = default_args,
    description       = "tanGo — recálculo diario de pesos estáticos (betweenness, pagerank)",
    start_date        = datetime(2026, 4, 26),
    schedule_interval = "0 3 * * *",   # 3am todos los días
    catchup           = False,
    tags              = ["tanGo", "pesos", "diario", "KAN-10"],
) as dag:

    inicio = EmptyOperator(task_id="inicio")
    fin    = EmptyOperator(task_id="fin")

    t_pesos    = PythonOperator(task_id="calcular_pesos",  python_callable=calcular_pesos)
    t_verificar = PythonOperator(task_id="verificar_pesos", python_callable=verificar_pesos)

    inicio >> t_pesos >> t_verificar >> fin