# -*- coding: utf-8 -*-
"""
DAG : Master_SA — Staging Area EventZella
==========================================
Orchestration du pipeline d'extraction et chargement dans la Staging Area.

Pipeline Talend d'origine : Master_SA_0.1.item
Source : TOS_DI-8.0.1 (Talend Open Studio)
Cible  : SA_Eventzella (SQL Server)

Structure :
  - Extraction sources métier (parallèle)
  - Web scraping enrichissement (parallèle)
  - Chargement Staging Area
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

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
    dag_id="eventzella_master_sa",
    default_args=default_args,
    description="Pipeline Staging Area EventZella — extraction sources + web scraping",
    schedule_interval="0 2 * * *",   # Chaque jour à 02h00
    catchup=False,
    max_active_runs=1,
    tags=["EventZella", "Staging", "ETL"],
)

# ── Fonctions simulant les jobs Talend ────────────────────────────
def run_talend_job(job_name: str, **context):
    """Simule l'exécution d'un job Talend et enregistre le log."""
    import time, logging
    logging.info(f"[Talend] Démarrage job : {job_name}")
    start = time.time()
    # En production : subprocess.run(["talend_job_runner.sh", job_name])
    time.sleep(1)
    elapsed = round(time.time() - start, 2)
    logging.info(f"[Talend] Job {job_name} terminé en {elapsed}s")
    return {"job": job_name, "status": "success", "duration_s": elapsed}

def check_sa_connection(**context):
    logging.info("Vérification connexion SA_Eventzella...")
    import logging
    logging.info("Connexion SA_Eventzella : OK")

import logging

# ── Tâches ────────────────────────────────────────────────────────

# Début
start = EmptyOperator(task_id="start_pipeline_sa", dag=dag)

# Vérification connexion
check_conn = PythonOperator(
    task_id="check_sa_connection",
    python_callable=check_sa_connection,
    dag=dag,
)

# ── Groupe 1 : Extraction sources métier (parallèle) ─────────────
t_beneficiary = PythonOperator(
    task_id="extract_Beneficiary",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Beneficiary"},
    dag=dag,
)
t_event = PythonOperator(
    task_id="extract_Event",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Event"},
    dag=dag,
)
t_reservation = PythonOperator(
    task_id="extract_Reservation",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Reservation"},
    dag=dag,
)
t_provider = PythonOperator(
    task_id="extract_Provider",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Provider"},
    dag=dag,
)
t_category = PythonOperator(
    task_id="extract_Category",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Category"},
    dag=dag,
)
t_subcategory = PythonOperator(
    task_id="extract_SubCategory",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "SubCategory"},
    dag=dag,
)
t_visitors = PythonOperator(
    task_id="extract_Visitors",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Visitors"},
    dag=dag,
)
t_complaint = PythonOperator(
    task_id="extract_Complaint",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Complaint"},
    dag=dag,
)
t_evaluation = PythonOperator(
    task_id="extract_Evaluation",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Evaluation"},
    dag=dag,
)
t_dates = PythonOperator(
    task_id="extract_Dates",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Dates"},
    dag=dag,
)
t_marketing = PythonOperator(
    task_id="extract_MarketingSpend",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "MarketingSpend"},
    dag=dag,
)

# ── Groupe 2 : Web scraping enrichissement (parallèle) ───────────
t_benchmarking = PythonOperator(
    task_id="scrape_BenchmarkingPrice",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Benchmarking_price_scrapped"},
    dag=dag,
)
t_holidays = PythonOperator(
    task_id="scrape_Holidays",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Holidays_scrapp"},
    dag=dag,
)
t_venues = PythonOperator(
    task_id="scrape_Venues",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Venues_scrapp"},
    dag=dag,
)
t_tendances = PythonOperator(
    task_id="scrape_Tendances",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "Tendances_Evennementielles_Scrapped"},
    dag=dag,
)

# ── Chargement final Staging Area ─────────────────────────────────
t_sa = PythonOperator(
    task_id="load_SA_Eventzella",
    python_callable=run_talend_job,
    op_kwargs={"job_name": "SA_Eventzella"},
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Fin
end = EmptyOperator(
    task_id="end_pipeline_sa",
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# ── Dépendances ───────────────────────────────────────────────────
sources_metier  = [t_beneficiary, t_event, t_reservation, t_provider,
                   t_category, t_subcategory, t_visitors, t_complaint,
                   t_evaluation, t_dates, t_marketing]
sources_scraping = [t_benchmarking, t_holidays, t_venues, t_tendances]

start >> check_conn >> sources_metier + sources_scraping >> t_sa >> end
