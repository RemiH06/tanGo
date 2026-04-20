# tanGo — Simulación visual

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
pip install plotly
python tests/sim1/tango_sim.py
```

Luego abre `tests/sim1/tango_sim.html` en tu navegador.

## Qué simula

Tres escenarios con el motor real de tanGo:

| Escenario | Hora | Condición |
|---|---|---|
| Hora pico | Lunes 8am | Soleado |
| Madrugada | Miércoles 2am | Despejado |
| Lluvia | Sábado 3pm | Lluvia |

## Qué clases del core usa

| Clase | Archivo | Para qué |
|---|---|---|
| `TrafficGraph` | `graph/simulator.py` | Grafo de 9 intersecciones |
| `WeightEngine` | `core/weight_engine.py` | Calcular presión por intersección |
| `TrafficContext` | `core/context.py` | Contexto ambiental de cada escenario |
| `Intersection.adjust_phase()` | `core/road.py` | Máquina de estados del semáforo |
| `Vehicle`, `Pedestrian` | `core/entities.py` | Entidades generadas por tick |

## Cómo leer el gráfico

- **Color del nodo** → fase del semáforo (verde/amarillo/rojo)
- **Tamaño del nodo** → presión acumulada (más grande = más demanda)
- **Número bajo el nodo** → presión numérica (rojo si ≥ 100 = cambio de fase)
- **Color de la arista** → tipo de vía (azul = av. principal, morado = secundaria, gris = calle)
- **Hover sobre un nodo** → detalles: fase, presión, conteo de entidades