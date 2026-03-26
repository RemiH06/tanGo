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
        
        Semáforo Inteligente de Tráfico         version 0.1.0-dev
        by Hex (@RemiH06), , ,
```

![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)
![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-yellow.svg?style=for-the-badge)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20DB-008CC1?style=for-the-badge)

---

## 🚦 Descripción general

**tanGo** es un sistema de ingeniería de datos que controla semáforos en tiempo real mediante un modelo de **pesos dinámicos** sobre un grafo vial. El nombre viene de **tan**gente (las funciones `sin/cos` que modelan ciclos temporales) y **Go** de ir. Como el baile: sincronizado, fluido, en tiempo real.

```diff
+ El sistema decide cuándo cambiar un semáforo según la presión acumulada de
+ vehículos y peatones en cada intersección, ajustada por clima, hora y día.
- No es un sistema de semáforos de tiempo fijo.
- No depende de un solo proveedor de datos.
```

---

## ⚙️ Arquitectura

```
tango/
├── core/
│   ├── context.py          # TrafficContext — dataclass inmutable por ciclo
│   ├── entities.py         # TrafficEntity (ABC), Vehicle, Pedestrian
│   ├── road.py             # RoadSegment, Intersection
│   └── weight_engine.py    # Motor de pesos — funciones puras
├── safety/
│   ├── guard.py            # SafetyGuard — silla de ruedas, emergencias, giros
│   └── circuit_breaker.py  # Resiliencia ante fallos de APIs externas
├── ingest/
│   └── base.py             # DataIngester (ABC), TomTomIngester, WeatherIngester
├── graph/
│   └── simulator.py        # CitySimulator — Neo4j + NetworkX
├── pipeline/               # Apache Airflow DAGs
├── api/                    # FastAPI — exposición de señales en tiempo real
└── tests/
    └── test_weight_engine.py
```

---

## 🧮 Sistema de pesos

Cada entidad en una intersección contribuye con un **peso** al total de presión. Cuando la presión supera el umbral de la vía, el semáforo cambia de fase.

```
Σ peso_efectivo(entidades) / peso_efectivo(vía) × 100 ≥ 100  →  cambio de fase
```

**Pesos base de entidades:**

| Entidad | Peso base |
|---|---|
| Peatón | 10 |
| Vehículo (auto) | 5 |
| Autobús | 8 |
| Bicicleta | 2 |
| Emergencia | 999 (override inmediato) |

**Pesos base de vías:**

| Vía | Peso base |
|---|---|
| Autopista / periférico | 100 |
| Avenida principal | 80 |
| Avenida secundaria | 50 |
| Calle residencial | 20 |
| Callejón | 5 |

**Modificadores dinámicos por contexto:**

| Condición | Efecto |
|---|---|
| 🌧️ Lluvia | Peatón × 1.3 |
| 🌡️ Temperatura extrema (< 5°C o > 35°C) | Peatón × 1.3 |
| 🌙 Madrugada (00:00 – 05:00) | Vehículo × 1.5 · Peatón × 0.8 |
| 📅 Fin de semana | Avenida × 0.7 (distribuir flujo) |
| ♿ Silla de ruedas | Verde mínimo extendido automáticamente |
| 🚨 Vehículo de emergencia | Override inmediato — todas las fases a rojo |

---

## 🏗️ Stack tecnológico

| Capa | Herramienta | Razón |
|---|---|---|
| Grafos (persistencia) | Neo4j + CQL + GDS + Bloom | Ciudad modelada como grafo nativo |
| Grafos (algoritmos) | NetworkX | Dijkstra, propagación ola verde |
| Procesamiento | Python 3.11, Pandas, NumPy | ETL + codificación trigonométrica temporal |
| Ingesta | httpx async | TomTom Traffic API + Open-Meteo |
| Orquestación | Apache Airflow | DAG cada 5 minutos, ciclo continuo |
| API | FastAPI + Uvicorn | Exposición de señales en tiempo real |
| Dashboard | Streamlit + Folium | Mapa de presión + métricas |
| Testing | pytest + pytest-asyncio | Funciones puras → tests sin mocks |
| Contenedores | Docker | Portabilidad del pipeline completo |

---

## 🔒 Seguridad

- API keys exclusivamente en variables de entorno — nunca en el código fuente
- `SecurityLayer`: validación de tokens, rate limiting, sanitización de inputs
- `CircuitBreaker`: si TomTom o Open-Meteo fallan, el pipeline continúa con fallback seguro
- Auditoría de eventos críticos (cambios de fase, overrides de emergencia)

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Los tests del `WeightEngine` son completamente deterministas — no requieren red ni base de datos porque todas las funciones son puras. Dado el mismo `TrafficContext`, siempre devuelven el mismo resultado.

---

## 🗂️ Variables de entorno

```env
TOMTOM_API_KEY=tu_api_key_aqui
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=tu_password_aqui
CITY_LATITUDE=20.6597
CITY_LONGITUDE=-103.3496
```

---

## 👥 Equipo

Proyecto Final — Ingeniería de Datos  
Instituto Tecnológico y de Estudios Superiores de Occidente (ITESO)  
Profesor: Moisés Flores Ortiz

---

## 📄 Licencia

Este proyecto está licenciado bajo la [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

> Cualquier modificación o uso en red debe publicarse bajo la misma licencia.