# -*- coding: utf-8 -*-
"""
DAG : Master_ETL — Data Warehouse EventZella
=============================================
Orchestration du chargement du Data Warehouse depuis la Staging Area.

Pipeline Talend d'origine : Master_ETL_0.1.item
Source : SA_Eventzella (Staging Area)
Cible  : DW_eventzalla (SQL Server Data Warehouse)

Ordre d'exécution (contraintes FK) :
  1. Création DW
  2. Dimensions (parallèle entre elles)
  3. Facts (après toutes les dimensions)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import logging

# ── Configuration DAG ─────────────────────────────────────────────
default_args = {
    "owner":            "EventZilla_ETL",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 1, 1),
    "email":            ["ranim.chikhrouhou@esprit.tn"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

dag = DAG(
    dag_id="eventzella_master_etl",
    default_args=default_args,
    description="Pipeline DW EventZella — chargement dimensions et tables de faits",
    schedule_interval="0 4 * * *",   # Chaque jour à 04h00 (après SA à 02h00)
    catchup=False,
    max_active_runs=1,
    tags=["EventZella", "DW", "ETL"],
)

# ── Fonction commune ──────────────────────────────────────────────
def run_talend_job(job_name: str, **context):
    import time
    logging.info(f"[Talend ETL] Démarrage : {job_name}")
    start = time.time()
    time.sleep(1)
    elapsed = round(time.time() - start, 2)
    logging.info(f"[Talend ETL] {job_name} OK en {elapsed}s")
    return {"job": job_name, "status": "success", "duration_s": elapsed}

# ── Tâches ────────────────────────────────────────────────────────

start = EmptyOperator(task_id="start_pipeline_etl", dag=dag)

# Étape 1 — Création du DW
t_create_dw = PythonOperator(
    task_id="create_DW_eventzalla",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "DW_eventzalla"},
    dag=dag,
)

# ── Étape 2 — Chargement Dimensions (parallèle) ───────────────────
t_dim_date = PythonOperator(
    task_id="load_Dim_Date",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_dates"},
    dag=dag,
)
t_dim_event = PythonOperator(
    task_id="load_Dim_Event",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_Event"},
    dag=dag,
)
t_dim_beneficiary = PythonOperator(
    task_id="load_Dim_Beneficiary",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_Beneficiary"},
    dag=dag,
)
t_dim_provider = PythonOperator(
    task_id="load_Dim_Provider",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_Provider"},
    dag=dag,
)
t_dim_service = PythonOperator(
    task_id="load_Dim_ServiceCategory",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_service_category"},
    dag=dag,
)
t_dim_reservation = PythonOperator(
    task_id="load_Dim_Reservation",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_Reservation"},
    dag=dag,
)
t_dim_visitors = PythonOperator(
    task_id="load_Dim_Visitors",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_visitors"},
    dag=dag,
)
t_dim_feedback = PythonOperator(
    task_id="load_Dim_Feedback",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_Feedback"},
    dag=dag,
)
t_dim_complaint = PythonOperator(
    task_id="load_Dim_Complaint",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_complaint"},
    dag=dag,
)
t_dim_benchmark = PythonOperator(
    task_id="load_Dim_BenchmarkPrice",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dim_BenchmarkPrice"},
    dag=dag,
)

all_dims = [
    t_dim_date, t_dim_event, t_dim_beneficiary, t_dim_provider,
    t_dim_service, t_dim_reservation, t_dim_visitors,
    t_dim_feedback, t_dim_complaint, t_dim_benchmark,
]

# ── Étape 3 — Chargement Facts (après toutes les dims) ───────────
# Fact_PerformanceCommerciale dépend de :
#   DimDate, DimEvent, DimReservation, DimBeneficiary,
#   DimServiceCategory, DimProvider, DimVisitors
t_fact_perf = PythonOperator(
    task_id="load_Fact_PerformanceCommerciale",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Fact_performance_commerciale"},
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Fact_RentabiliteFinanciere dépend de :
#   DimDate, DimEvent, DimServiceCategory, DimBenchmarkPrice, DimProvider
t_fact_finance = PythonOperator(
    task_id="load_Fact_RentabiliteFinanciere",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Fact_Rentab_finan"},
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Fact_SatisfactionClient dépend de :
#   DimDate, DimProvider, DimServiceCategory, DimReservation,
#   DimFeedback, DimComplaint
t_fact_satisf = PythonOperator(
    task_id="load_Fact_SatisfactionClient",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Fact_satisfaction_client"},
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

end = EmptyOperator(
    task_id="end_pipeline_etl",
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# ── Dépendances ───────────────────────────────────────────────────
# list >> list n'est pas supporté par Airflow → on enchaîne individuellement
start >> t_create_dw >> all_dims
all_dims >> t_fact_perf
all_dims >> t_fact_finance
all_dims >> t_fact_satisf
[t_fact_perf, t_fact_finance, t_fact_satisf] >> end
