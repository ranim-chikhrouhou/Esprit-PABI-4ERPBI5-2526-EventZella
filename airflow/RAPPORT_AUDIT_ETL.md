# Rapport d'Audit ETL — Apache Airflow / Talend
## Projet EventZella — Critère E : ETL & Pipeline Audit

---

## 1. Architecture du Pipeline ETL

### Outil ETL : Talend Open Studio 8.0.1
Le pipeline ETL EventZella est développé avec **Talend Open Studio for Data Integration (TOS_DI-8.0.1)**, organisé en 3 projets Master :

| Projet Talend | Rôle | Planification Airflow |
|---|---|---|
| `Master_SA` | Extraction + Staging Area | Quotidien à 02h00 |
| `Master_ETL` | Chargement Data Warehouse | Quotidien à 04h00 |
| `Master_Global` | Orchestrateur complet SA → DW | Hebdomadaire (dimanche 01h00) |

---

## 2. Structure des DAGs Airflow

### DAG 1 : `eventzella_master_sa` — Staging Area

```
start_pipeline_sa
       │
check_sa_connection
       │
       ├── [PARALLÈLE — Sources métier]
       │     ├── extract_Beneficiary
       │     ├── extract_Event
       │     ├── extract_Reservation
       │     ├── extract_Provider
       │     ├── extract_Category
       │     ├── extract_SubCategory
       │     ├── extract_Visitors
       │     ├── extract_Complaint
       │     ├── extract_Evaluation
       │     ├── extract_Dates
       │     └── extract_MarketingSpend
       │
       ├── [PARALLÈLE — Web Scraping]
       │     ├── scrape_BenchmarkingPrice
       │     ├── scrape_Holidays
       │     ├── scrape_Venues
       │     └── scrape_Tendances
       │
load_SA_Eventzella
       │
end_pipeline_sa
```

**15 tâches | Temps estimé : ~12 minutes | Retries : 2**

---

### DAG 2 : `eventzella_master_etl` — Data Warehouse

```
start_pipeline_etl
       │
create_DW_eventzalla
       │
       ├── [PARALLÈLE — 10 Dimensions]
       │     ├── load_Dim_Date
       │     ├── load_Dim_Event
       │     ├── load_Dim_Beneficiary
       │     ├── load_Dim_Provider
       │     ├── load_Dim_ServiceCategory
       │     ├── load_Dim_Reservation
       │     ├── load_Dim_Visitors
       │     ├── load_Dim_Feedback
       │     ├── load_Dim_Complaint
       │     └── load_Dim_BenchmarkPrice
       │
       ├── [PARALLÈLE — 3 Tables de Faits]
       │     ├── load_Fact_PerformanceCommerciale
       │     │     └── dépend de : Date, Event, Reservation, Beneficiary,
       │     │                     ServiceCategory, Provider, Visitors
       │     ├── load_Fact_RentabiliteFinanciere
       │     │     └── dépend de : Date, Event, ServiceCategory, Benchmark, Provider
       │     └── load_Fact_SatisfactionClient
       │           └── dépend de : Date, Provider, ServiceCategory,
       │                           Reservation, Feedback, Complaint
       │
end_pipeline_etl
```

**15 tâches | Temps estimé : ~8 minutes | Retries : 2**

---

### DAG 3 : `eventzella_master_global` — Orchestrateur Global

```
start_global_pipeline
       │
trigger_Master_SA ──(attend succès)──► trigger_Master_ETL
       │
end_global_pipeline
```

**Pipeline complet : ~20 minutes | Retry : 1**

---

## 3. Analyse des Dépendances & FK

Les dépendances entre tâches respectent les contraintes de clés étrangères définies dans le DW :

| Table de fait | Dimensions requises |
|---|---|
| `Fact_PerformanceCommerciale` | DimDate, DimEvent, DimReservation, DimBeneficiary, DimServiceCategory, DimProvider, DimVisitors |
| `Fact_RentabiliteFinanciere` | DimDate, DimEvent, DimServiceCategory, DimBenchmarkPrice, DimProvider |
| `Fact_SatisfactionClient` | DimDate, DimProvider, DimServiceCategory, DimReservation, DimFeedback, DimComplaint |

**→ Les Facts ne s'exécutent qu'après le succès de toutes les Dimensions** (`trigger_rule=ALL_SUCCESS`)

---

## 4. Gestion des Erreurs & Robustesse

### Politique de retry
- **Master_SA** : 2 retries avec délai de 5 minutes (scraping instable par nature)
- **Master_ETL** : 2 retries avec délai de 5 minutes
- **Master_Global** : 1 retry avec délai de 10 minutes

### Alertes email
- Envoi automatique à `ranim.chikhrouhou@esprit.tn` en cas d'échec
- Configuration : `email_on_failure=True`

### Règles de déclenchement
- `ALL_SUCCESS` sur les Facts → garantit l'intégrité référentielle
- `ALL_DONE` sur les tâches finales → enregistre toujours la fin, même en cas d'erreur partielle

---

## 5. Planification & Scheduling

| DAG | Cron | Fréquence | Fenêtre d'exécution |
|---|---|---|---|
| `eventzella_master_sa` | `0 2 * * *` | Quotidien | 02h00–04h00 |
| `eventzella_master_etl` | `0 4 * * *` | Quotidien | 04h00–06h00 |
| `eventzella_master_global` | `0 1 * * 0` | Hebdomadaire | Dimanche 01h00 |

La planification assure que :
1. Le Staging Area est prêt avant le début de l'ETL (fenêtre de 2h)
2. Le pipeline complet du dimanche démarre avant les runs journaliers

---

## 6. Logs & Traçabilité

Apache Airflow enregistre automatiquement :
- **Logs par tâche** : `/opt/airflow/logs/<dag_id>/<task_id>/<execution_date>/`
- **Durée d'exécution** : visible dans le Gantt chart de l'interface web
- **Statut de chaque run** : success / failed / skipped / upstream_failed
- **Historique complet** : accessible via `http://localhost:8080`

---

## 7. Correspondance Talend ↔ Airflow

| Job Talend | Tâche Airflow | DAG |
|---|---|---|
| `SA_Eventzella` | `load_SA_Eventzella` | master_sa |
| `Beneficiary` | `extract_Beneficiary` | master_sa |
| `Event` | `extract_Event` | master_sa |
| `Reservation` | `extract_Reservation` | master_sa |
| `Provider` | `extract_Provider` | master_sa |
| `Benchmarking_price_scrapped` | `scrape_BenchmarkingPrice` | master_sa |
| `Holidays_scrapp` | `scrape_Holidays` | master_sa |
| `DW_eventzalla` | `create_DW_eventzalla` | master_etl |
| `Dim_dates` | `load_Dim_Date` | master_etl |
| `Dim_BenchmarkPrice` | `load_Dim_BenchmarkPrice` | master_etl |
| `Fact_performance_commerciale` | `load_Fact_PerformanceCommerciale` | master_etl |
| `Fact_Rentab_finan` | `load_Fact_RentabiliteFinanciere` | master_etl |
| `Fact_satisfaction_client` | `load_Fact_SatisfactionClient` | master_etl |
| `Master_SA` | `trigger_Master_SA` | master_global |
| `Master_ETL` | `trigger_Master_ETL` | master_global |
