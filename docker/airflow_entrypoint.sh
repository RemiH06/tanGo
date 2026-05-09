#!/bin/bash
set -e

echo "tanGo — Inicializando Airflow..."

# Inicializar base de datos si no existe
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "  Creando base de datos Airflow..."
    airflow db init

    echo "  Creando usuario admin..."
    airflow users create \
        --username admin \
        --password admin \
        --firstname tanGo \
        --lastname Admin \
        --role Admin \
        --email admin@tango.local
fi

echo "  Iniciando Airflow standalone..."
exec airflow standalone