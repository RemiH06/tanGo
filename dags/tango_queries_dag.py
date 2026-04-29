from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

# Configuración base del DAG
default_args = {
    'owner': 'diego',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}

# Definición del DAG
with DAG(
    dag_id='tango_traffic_graph_pipeline',
    default_args=default_args,
    description='Pipeline para extraer pesos y nodos del sistema de tráfico',
    start_date=datetime(2026, 4, 26),
    schedule_interval='@hourly', # Corremos cada hora para actualizar los semáforos
    catchup=False,
    tags=['tanGo', 'grafos', 'trafico', 'jira-task'],
) as dag:

    inicio = EmptyOperator(task_id='inicio')

    # 1. Extraer el estado actual de la red vial (nodos, grados y pesos)
    extraer_red_vial = SQLExecuteQueryOperator(
        task_id='extraer_pesos_calles',
        conn_id='tango_db_conn',
        sql="""
            SELECT 
                interseccion_origen_id, 
                interseccion_destino_id, 
                distancia_metros, 
                grado_interseccion,
                tiempo_base_semaforo
            FROM red_vial_tango
            WHERE estado_calle = 'activa';
        """
    )

    # 2. Transformación/Limpieza: Quitar registros basura o nodos desconectados
    limpiar_nodos = SQLExecuteQueryOperator(
        task_id='limpiar_nodos_huerfanos',
        conn_id='tango_db_conn',
        sql="""
            DELETE FROM red_vial_tango 
            WHERE grado_interseccion < 1 OR interseccion_destino_id IS NULL;
        """
    )

    fin = EmptyOperator(task_id='fin')

    # Definimos el flujo (dependencias)
    inicio >> extraer_red_vial >> limpiar_nodos >> fin