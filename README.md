![Made with Python](https://forthebadge.com/images/badges/made-with-python.svg)
![Built by Developers](http://ForTheBadge.com/images/badges/built-by-developers.svg)
![Uses Git](http://ForTheBadge.com/images/badges/uses-git.svg)
![Built with Love](http://ForTheBadge.com/images/badges/built-with-love.svg)

```
████████╗ █████╗ ███╗   ██╗ ██████╗  ██████╗ 
╚══██╔══╝██╔══██╗████╗  ██║██╔════╝ ██╔═══██╗
   ██║   ███████║██╔██╗ ██║██║  ███╗██║   ██║
   ██║   ██╔══██║██║╚██╗██║██║   ██║██║   ██║
   ██║   ██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ 
        
        Semáforo Inteligente de Tráfico         version 1.0.0
        by Hex (@RemiH06), @cesarsantos23, @edumar67, @DonCo93
```

---

## 🚦 Descripción general

**tanGo** es un sistema de coordinación inteligente de semáforos que combina un algoritmo inspirado en **SCOOT** (Split Cycle Offset Optimisation Technique) con **Reinforcement Learning** (PPO) para minimizar el tiempo de detención en intersecciones del centro de Guadalajara.

El nombre viene de **tan**gente (las funciones `sin/cos` que modelan ciclos temporales) y **Go** de ir. Como el baile: sincronizado, fluido, en tiempo real.

El sistema opera sobre un grafo real de **354 intersecciones y 672 segmentos viales** extraído de OpenStreetMap. Cada hora, un DAG de Airflow consulta datos reales de clima (Open-Meteo) y velocidades de tráfico (TomTom), corre 15 ticks de simulación y exporta el estado al dashboard.

```diff
+ Agente PPO entrenado: 15× menos vehículos detenidos que el baseline
+ 1,036 vehículos completaron su ruta en 60 ticks (vs 0 del baseline)
+ Adaptable a cualquier ciudad con datos en OpenStreetMap
- No es un sistema de semáforos de tiempo fijo
- No depende de un solo proveedor de datos
```

> 📖 **Documentación completa:** [remih06.github.io/tanGo](https://remih06.github.io/tanGo)

---

## ⚙️ Arquitectura

```
36.TanGo/
├── core/
│   ├── algorithm.py        # TrafficAlgorithm — 3 pasos secuenciales
│   ├── context.py          # TrafficContext — hora, clima, tráfico real
│   ├── entities.py         # Vehicle, Pedestrian, VehicleType
│   ├── road.py             # Intersection, Phase, RoadSegment
│   ├── movement.py         # MovementEngine — Dijkstra + velocidades
│   └── weight_engine.py    # Presión, ola verde, pesos estáticos
├── graph/
│   ├── city_graph.json     # 354 nodos, 672 aristas — GDL centro
│   ├── tango_state.json    # Estado horario generado por DAG
│   └── city_loader.py      # Overpass API → JSON + pesos estáticos
├── dags/
│   ├── tango_queries_dag.py   # Pipeline @hourly
│   └── tango_daily_dag.py     # Recálculo de pesos @3am
├── dashboard/
│   ├── api.py              # FastAPI :8000
│   ├── app.py              # Streamlit :8501
│   └── tanGo_dashboard.html   # Dashboard principal — modo claro/oscuro
├── tests/
│   ├── sim0/               # Baseline: timers fijos
│   ├── sim1/               # SCOOT greedy
│   ├── sim2/               # Pesos estáticos + MovementEngine
│   └── sim3/               # PPO — agente RL entrenado
│       ├── tango_env.py    # TanGoEnv (gymnasium)
│       ├── train.py
│       ├── evaluate.py
│       └── tango_sim3.py   # Visualización del agente PPO
├── YOLO_PID/
│   ├── semaforo_detector.py   # Detección YOLO + eventos Kafka
│   ├── requirements.txt       # ultralytics, opencv, kafka-python
│   └── setup.bat              # Setup automático Windows
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.airflow
│   └── Dockerfile.dashboard
├── docker-compose.yml
└── docs/
    └── index.html          # Documentación interactiva
```

---

## 🧮 El algoritmo — 3 pasos por tick

Cada tick (~30 segundos simulados), `TrafficAlgorithm.run_tick()` ejecuta:

**Paso 1 — Presión propia**
`WeightEngine.aggregate_pressure()` calcula una presión escalar por intersección a partir de las entidades presentes. Función pura: mismo input → mismo output.

**Paso 2 — Mente colmena**
Cada nodo recibe señales de sus vecinos upstream. Si un vecino está en verde, se calcula el offset temporal (distancia / velocidad) para anticipar la llegada del flujo. Cuando el offset se cumple, el nodo downstream se fuerza a verde — esto implementa la **ola verde distribuida** de SCOOT.

**Paso 3 — Ajuste de fases**
Aplica la máquina de estados (RED → GREEN → YELLOW → RED) con exclusión mutua NS/EW, timeout de equidad, modo BLINK para intersecciones vacías y coordinación de cluster (intersecciones a <60m se coordinan entre sí).

---

## 🤖 Reinforcement Learning — sim3

El agente PPO observa **10 features por semáforo** (presión, fase, entidades, hora cíclica sin/cos, presión de vecinos upstream, wave_offset) y decide en cada tick si mantener o cambiar cada fase.

**Resultados tras 550 episodios de entrenamiento (~5 días en CPU):**

| Métrica | sim0 (baseline) | sim1 (SCOOT) | sim3 (PPO) |
|---|---|---|---|
| Detenidos/tick | 367 | 491 | **24.18** |
| Vehículos llegados | 0 | 0 | **1,036** |
| Reducción vs baseline | — | — | **−93%** |

> sim3b (multi-agente) queda como trabajo futuro — con los recursos actuales (i3-11va, 20GB RAM) es inviable: sim3 requirió ~500,000 segundos de CPU para 550 episodios. Un esquema multi-agente multiplicaría ese costo por el número de agentes.

---

## 🏗️ Stack tecnológico

| Capa | Herramienta | Versión |
|---|---|---|
| Algoritmo | NetworkX | 3.3 |
| RL entrenamiento | Ray RLlib | 2.55 |
| RL entorno | Gymnasium | 0.29 |
| RL modelo | PyTorch | 2.1+ |
| Pipeline | Apache Airflow | 2.9 |
| API | FastAPI + Uvicorn | 0.111 / 0.30 |
| Dashboard | Streamlit | 1.35 |
| Visualización | Leaflet.js | 1.9.4 |
| Object detection | YOLOv8 (ultralytics) | 8.3.0 |
| Streaming | Kafka + kafka-python | 2.0.2 |
| Datos externos | Overpass, Open-Meteo, TomTom | — |
| Contenedores | Docker + Compose | 29.2 |

---

## 🚀 Quickstart con Docker

```bash
git clone https://github.com/RemiH06/tanGo.git
cd tanGo
docker compose up --build
```

| Servicio | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Airflow | http://localhost:8082 (admin/admin) |
| Streamlit | http://localhost:8501 |

Trigger manual del DAG para poblar el estado inicial:

```bash
docker compose exec tango-airflow airflow dags trigger tango_traffic_pipeline
```

---

## 🖥️ Quickstart local (WSL / Linux)

```bash
# Entorno del algoritmo y visualizaciones
python -m venv ~/tango_env
source ~/tango_env/bin/activate
pip install -r requirements_sim3.txt

# Generar visualización del agente PPO
python tests/sim3/tango_sim3.py --checkpoint tests/sim3/checkpoints/checkpoint_00550

# Entorno de Airflow + FastAPI
python -m venv ~/airflow_env
source ~/airflow_env/bin/activate
pip install -r requirements_airflow.txt

# Levantar Airflow
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8082
airflow standalone

# Levantar FastAPI (otra terminal)
uvicorn dashboard.api:app --reload --port 8000
```

---

## 🗂️ Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
TOMTOM_API_KEY=tu_api_key_aqui
CITY_LATITUDE=20.6597
CITY_LONGITUDE=-103.3496
CITY_RADIUS_M=800
```

Open-Meteo no requiere API key. TomTom es opcional — sin key el pipeline usa factor de tráfico 1.0 (flujo libre).

---

## 🔒 Seguridad

- API keys exclusivamente en variables de entorno — nunca en el código fuente
- `CircuitBreaker`: si TomTom o Open-Meteo fallan, el pipeline continúa con fallback seguro
- Un solo escritor (`tango_state.json`) — múltiples lectores sin condiciones de carrera

---

## 🧪 Tests

```bash
source ~/airflow_env/bin/activate
pytest tests/ -v
```

Los tests del `WeightEngine` son completamente deterministas — no requieren red ni base de datos porque todas las funciones son puras.

---

## 📊 KANs completados

- ✅ sim0 — Baseline timers fijos
- ✅ sim1 — TrafficAlgorithm + SCOOT greedy + casos experimentales
- ✅ sim2 — Pesos estáticos + MovementEngine + Dijkstra
- ✅ sim3 — TanGoEnv + PPO (Ray 2.55) + evaluación comparativa
- ✅ KAN-10 — DAG Airflow (@hourly + @daily)
- ✅ KAN-11 — FastAPI + Dashboard HTML técnico
- ✅ KAN-12 — Docker (3 servicios)
- ✅ KAN-15 — Documentación interactiva GitHub Pages
- ✅ KAN-16 — VisionIngester / YOLO (semaforo_detector.py)
- ✅ KAN-17 — Kafka (integrado en semaforo_detector.py, pendiente de producción)

---

## 📷 VisionIngester — YOLO + Kafka

`YOLO_PID/semaforo_detector.py` detecta vehículos detenidos en video usando YOLOv8 y publica eventos a Kafka cuando un vehículo lleva más de 2 segundos quieto.

**Setup (Windows):**

```bat
cd YOLO_PID
setup.bat
```

**Correr manualmente:**

```bash
cd YOLO_PID
python semaforo_detector.py
```

**Configuración en `semaforo_detector.py`:**

```python
VIDEO_PATH     = r"ruta\al\video.mp4"   # fuente de video
OUTPUT_PATH    = r"ruta\output.mp4"     # video anotado
KAFKA_BROKER   = "localhost:9092"        # broker Kafka
KAFKA_TOPIC    = "semaforo-eventos"      # topic de eventos
```

**Eventos Kafka emitidos:**

```json
{
  "evento":     "objeto_quieto",
  "objeto_id":  42,
  "clase":      "car",
  "posicion":   {"x": 320, "y": 240},
  "seg_quieto": 3.5,
  "timestamp":  1715000000.123,
  "accion":     "cambiar_verde"
}
```

Kafka es opcional — si no hay broker disponible, el detector corre con `kafka_ok=False` y solo genera el video anotado sin enviar eventos.

> **Nota:** En el estado actual del proyecto, el DAG de Airflow y el detector YOLO corren de forma independiente. La integración completa (Kafka → DAG → algoritmo) está diseñada para producción.

---

## 👥 Equipo

Proyecto Final — Ingeniería de Datos  
Instituto Tecnológico y de Estudios Superiores de Occidente (ITESO)  
Profesor: Moisés Flores Ortiz

---

## 📄 Licencia

Este proyecto está licenciado bajo la [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

> Cualquier modificación o uso en red debe publicarse bajo la misma licencia.