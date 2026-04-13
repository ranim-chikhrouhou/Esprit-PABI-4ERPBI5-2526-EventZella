# Machine Learning — EventZilla



Workflow : **préparation** → **4 familles de modèles** → **évaluation comparative**.



## Organisation du dossier `ML/` (partage équipe)



**Important — chemins figés dans le code :** ne **renommez pas** et ne **déplacez pas** sans mise à jour du code : `ml_paths.py`, `processed/`, `models_artifacts/`, le package Python `ML/` (modules à la racine de ce dossier).



| Emplacement | Rôle |

|-------------|------|

| **`ml_paths.py`** | Définit `ML_PROCESSED` → `ML/processed/`, `ML_MODELS` → `ML/models_artifacts/`, connexion SQL. |

| **`processed/`** | Données et artefacts de préparation (parquet, `.npy`, JSON, figures préfixées `A_*`). |

| **`models_artifacts/`** | Pipelines `.joblib`, `metrics_*.json` produits par les notebooks **01–04**. |

| **`notebooks/`** | Chaîne d’analyse `00_` … `05_*.ipynb` ; Jupyter doit être lancé depuis la **racine du dépôt** (dossier parent de `ML/`). |

| **`scripts/`** | Pipelines `run_*.py`, `run_all_ml_pipeline.py`, `generate_notebooks.py`. |

| **`schema_eventzilla.py`** | Requêtes SQL alignées sur le DW. |

| **`EventZilla_Dashboards_Improved.pdf`** | Référence KPI (chemins relatifs `ML/...` dans certains scripts). |



Arborescence logique :



```text

ML/

├── README.md                 ← ce fichier

├── requirements.txt          ← dépendances Python (dont jinja2 pour pandas.Styler dans le notebook 05)

├── ml_paths.py               ← chemins et SQL (ne pas déplacer)

├── schema_eventzilla.py

├── EventZilla_Dashboards_Improved.pdf

├── notebooks/                ← 00_A_*.ipynb … 05_*.ipynb

├── scripts/

│   ├── run_*.py

│   ├── run_all_ml_pipeline.py

│   ├── run_test_sql_connection.py

│   └── generate_notebooks.py

├── processed/                ← généré par 00_A (ne pas renommer)

├── models_artifacts/         ← généré par 01–04 (ne pas renommer)

├── ML_METRICS_SUMMARY.md     ← généré par 05 (optionnel au partage)

└── (scripts Python utilitaires à la racine de ML/ : enrichissement / réorganisation — optionnels)

```



## Fichiers locaux (Excel / CSV) — fallback ML



Sous **`ML/`** (hors `processed/`, `models_artifacts/`, `notebooks/`, `scripts/`, `__pycache__`) : **`.xlsx`**, **`.xls`**, **`.csv`**. Exemple typique : **`Reservation.xlsx`** (`id_reservation`, `status`, `final_price`, `reservation_date`, …). Dépendances : `openpyxl` (xlsx), `xlrd` (xls) — voir `config/ml-requirements.txt`. Peuvent compléter **FilesMachine/data_original/** et **datascrapped/**.



Si les jointures SQL échouent sur **`DimReservation`**, définir par exemple :  

`EVENTZILLA_DIM_RESERVATION_PK` / `EVENTZILLA_FACT_RESERVATION_FK` (noms de colonnes réels côté SQL).



## Données : DW restauré (recommandé)



Les sauvegardes dans **`FilesMachine/DB/`** :



- **`DW_Eventzilla`** → restaurer en base SQL Server (nom conseillé : `DW_Eventzilla`)

- **`SA_eventzilla`** → staging (optionnel pour l’ETL ; le ML utilise surtout le **DW**)



Schéma des tables / jointures : **`ML/schema_eventzilla.py`** (aligné sur `ScriptsDiagrams/EventZilla_DWH_Par_Fact_Mermaid.md` et les noms Power BI : `Fact_RentabiliteFinanciere`, `Fact_PerformanceCommerciale`, `DimReservation`, `DimDate`, …).



## Connexion Python (Windows Authentication)



Le projet est maintenant préconfiguré par défaut pour :



- **Server :** `ASUSRANIM`

- **Database :** `DW_eventzella`

- **Auth :** Windows (`Trusted_Connection=yes`)



Donc vous pouvez lancer les notebooks **sans définir d’URI**.



Si vous voulez surcharger :



```powershell

$env:EVENTZILLA_SQL_SERVER = "ASUSRANIM"

$env:EVENTZILLA_SQL_DW = "DW_eventzella"

# optionnel si port non standard :

$env:EVENTZILLA_SQL_PORT = "1433"

```



Ou fournir une URI complète :



```powershell

$env:EVENTZILLA_SQL_URI = "mssql+pyodbc://@ASUSRANIM/DW_eventzella?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes"

```



## Prérequis



```powershell

cd chemin\vers\PI BI NEW

pip install -r ML/requirements.txt

```



*(Alternative : `pip install -r config/ml-requirements.txt` si votre projet utilise ce fichier.)*



## Exécution sans Jupyter (Cursor / VS Code)



Dans le **terminal intégré**, répertoire courant = racine `PI BI NEW` :



```powershell

python ML/scripts/run_test_sql_connection.py

python ML/scripts/run_00_data_preparation.py

python ML/scripts/run_01_clustering.py

python ML/scripts/run_02_classification.py

python ML/scripts/run_03_prediction_regression.py

python ML/scripts/run_04_time_series.py

python ML/scripts/run_05_metrics_comparison.py

```



**Tout en une fois** (test SQL + étapes 00→05) :



```powershell

python ML/scripts/run_all_ml_pipeline.py

```



Ignorer le test SQL si besoin : `python ML/scripts/run_all_ml_pipeline.py --skip-connection-test`



Les notebooks `.ipynb` restent une documentation optionnelle ; le code source de vérité pour le terminal est les fichiers `run_*.py` sous `ML/scripts/`.



### Lancer un fichier Python dans VS Code / Cursor



- Ouvrir le fichier `ML/scripts/run_00_data_preparation.py`

- Clic droit → **Run Python File in Terminal** (ou bouton ▶︎ en haut à droite si l’extension Python est installée)



Le répertoire de travail du terminal doit être la **racine du dépôt** pour que les imports `ML.*` fonctionnent.



---



## Jupyter (optionnel)



Lancer Jupyter avec le répertoire courant = **racine du dépôt** (dossier qui contient `ML/`). Ouvrir les notebooks sous **`ML/notebooks/`**.



## Notebooks (objectifs = KPI *EventZilla Improved*)



Chaque notebook commence par un **objectif métier** et les **KPIs** associés (`docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`, copie PDF `ML/EventZilla_Dashboards_Improved.pdf`).



| Fichier (grille d’évaluation) | Objectif | Prétraitement / modèles clés |

|-------------------------------|----------|------------------------------|

| `notebooks/00_A_preparation_donnees_feature_engineering.ipynb` | **A** — Matrice features (DW) | Imputation, `StandardScaler`, `MinMaxScaler`, `X_raw_numeric.npy` |

| `notebooks/01_E_clustering_segmentation.ipynb` | **E** — Segments / diversité d’offre | K-Means vs agglomératif, coude, silhouette, PCA |

| `notebooks/02_C_classification_statut_reservation.ipynb` | **C** — Statut (acceptation / annulation) | RF vs régression logistique, GridSearchCV |

| `notebooks/03_D_regression_montants_KPI.ipynb` | **D** — Montants (`final_price`, …) | Ridge vs Random Forest, K-Fold |

| `notebooks/04_F_series_temporelles.ipynb` | **F** — Anticipation volume / CA mensuel | ADF/KPSS, Holt vs ARIMA |

| `notebooks/05_synthese_metriques_validation.ipynb` | Synthèse `metrics_*.json` | Tableau récapitulatif, export Markdown |



Régénérer les `.ipynb` après modification des gabarits :



```powershell

python ML/scripts/generate_notebooks.py

```



## Artéfacts



- `ML/processed/` — `dw_financial_wide.parquet`, `features_matrix.*`, `X_*.npy`

- `ML/models_artifacts/` — `.joblib`, `metrics_*.json`



## Interface web (Streamlit)



Application locale pour que **les enseignants (et l’équipe)** puissent **parcourir les métriques** des modèles (`metrics_*.json`) et **lancer des prédictions d’essai** sur des entrées cohérentes avec le **data warehouse** (sans écrire de SQL), à partir des artefacts dans `models_artifacts/`. Interface Plotly, charte **bleu profond / cyan** façon dashboard EventZilla.



Prérequis : `pip install -r ML/requirements.txt` (inclut `streamlit`, `plotly`).



Depuis la **racine du dépôt** :



```powershell

streamlit run ML/streamlit_app.py

```



Le fichier `.streamlit/config.toml` à la racine du projet applique le thème clair. Logo : `ML/assets/eventzilla_ticket.svg` (par défaut) ou placez **`ML/assets/eventzilla_logo.png`** pour utiliser votre fichier rond officiel. Les prédictions s’appuient sur `ML/processed/dw_financial_wide.parquet` pour les profils ; sans fichiers modèle, l’app affiche tout de même les métriques JSON.



## Références



- `FilesMachine/README.md` — restauration des fichiers `DW_Eventzilla` / `SA_eventzilla`

- `docs/Liste_Des_Kpis_Updated_English_DAX.md` — KPIs alignés dashboards

- `EDAs/` — EDA CSV / scraping

