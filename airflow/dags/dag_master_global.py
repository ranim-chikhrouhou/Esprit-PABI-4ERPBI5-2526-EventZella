# -*- coding: utf-8 -*-
"""
DAG : Master_Global — Pipeline Complet EventZella
==================================================
Orchestration du pipeline ETL complet : Staging Area → Data Warehouse.

Pipeline Talend d'origine : Master_Global_0.1.item
Enchaîne Master_SA puis Master_ETL via des TriggerDagRunOperator.

Planification : Chaque dimanche à 01h00 (run hebdomadaire complet)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.trigger_rule import TriggerRule

# ── Configuration DAG ─────────────────────────────────────────────
default_args = {
    "owner":            "EventZilla_ETL",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 1, 1),
    "email":            ["ranim.chikhrouhou@esprit.tn"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
}

dag = DAG(
    dag_id="eventzella_master_global",
    default_args=default_args,
    description="Pipeline ETL complet EventZella — SA + DW (orchestrateur global)",
    schedule_interval="0 1 * * 0",   # Chaque dimanche à 01h00
    catchup=False,
    max_active_runs=1,
    tags=["EventZella", "Global", "ETL"],
)

# ── Tâches ────────────────────────────────────────────────────────

start = EmptyOperator(
    task_id="start_global_pipeline",
    dag=dag,
)

# Étape 1 — Déclencher Master_SA
trigger_sa = TriggerDagRunOperator(
    task_id="trigger_Master_SA",
    trigger_dag_id="eventzella_master_sa",
    wait_for_completion=True,
    poke_interval=30,
    allowed_states=["success"],
    failed_states=["failed"],
    dag=dag,
)

# Étape 2 — Attendre la fin de Master_SA puis déclencher Master_ETL
trigger_etl = TriggerDagRunOperator(
    task_id="trigger_Master_ETL",
    trigger_dag_id="eventzella_master_etl",
    wait_for_completion=True,
    poke_interval=30,
    allowed_states=["success"],
    failed_states=["failed"],
    dag=dag,
)

end = EmptyOperator(
    task_id="end_global_pipeline",
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# ── Dépendances ───────────────────────────────────────────────────
start >> trigger_sa >> trigger_etl >> end
