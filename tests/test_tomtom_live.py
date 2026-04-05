"""
tests/test_tomtom_live.py
--------------------------
Prueba REAL contra la API de TomTom.
Este archivo NO es parte del test suite normal.

Cómo ejecutar desde la raíz del proyecto:
    python tests/test_tomtom_live.py
"""

import asyncio
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.context import TrafficContext
from ingest.tomtom import TomTomIngester
from safety.circuit_breaker import CircuitBreaker


async def main():
    print("=" * 55)
    print("  tanGo — prueba en vivo TomTom Traffic API")
    print("=" * 55)

    key = os.getenv("TOMTOM_API_KEY", "")
    if not key:
        print("\n[ERROR] TOMTOM_API_KEY no encontrada en .env")
        return
    print(f"\n  API key encontrada: {'*' * 28}{key[-4:]}")

    breaker  = CircuitBreaker(name="tomtom-live", failure_threshold=3)
    ingester = TomTomIngester(circuit_breaker=breaker)

    # Usar build() para que infiera hora pico, fin de semana, etc.
    ctx = TrafficContext.build(
        timestamp      = datetime.now(),
        temperature_c  = 22.0,
        is_raining     = False,
        wind_speed_kmh = 10.0,
        visibility_m   = 10000.0,
    )

    print(f"\n  Hora local     : {ctx.timestamp.strftime('%H:%M:%S')}")
    print(f"  Fin de semana  : {ctx.is_weekend}")
    print(f"  Hora pico      : {ctx.is_rush_hour}")
    print(f"  Madrugada      : {ctx.is_late_night}")
    print(f"\n  Consultando tráfico en Guadalajara...")

    try:
        # Llamar fetch() directamente para ver el error real si lo hay
        raw       = await ingester.fetch(ctx)
        snapshots = ingester.parse(raw)

        print("\n  Respuesta de TomTom:")
        print("  " + "-" * 45)
        for snap in snapshots:
            pct = snap.congestion_index * 100
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"  Segmento   : {snap.segment_id}")
            print(f"  Velocidad  : {snap.current_speed_kmh:.1f} km/h  (libre: {snap.free_flow_kmh:.1f} km/h)")
            print(f"  Congestión : [{bar}] {pct:.1f}%")
            print(f"  Confianza  : {snap.confidence * 100:.0f}%")

        print(f"\n  CircuitBreaker : {breaker.state.name}")
        print("  Prueba completada exitosamente.")

    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
        print("\n  Traceback completo:")
        traceback.print_exc()
        print(f"\n  CircuitBreaker : {breaker.state.name}")

    finally:
        await ingester.close()

    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())