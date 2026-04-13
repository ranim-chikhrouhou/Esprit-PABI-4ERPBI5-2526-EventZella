# EventZilla — Synthèse métriques ML et modèles champions

*Généré : 2026-04-13 02:13*

## Tableau comparatif (synthèse E, C, D, F)

| Critère | Domaine | Cible (Y) | Champion | Benchmark | Règle de choix | Qualité (synthèse) | KPI | Fichier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | Clustering | k=3 clusters (features standardisées) | KMeans | AgglomerativeClustering_ward | Silhouette (holdout) + Davies-Bouldin | Silh.=0.1508 / DB_K=1.823 / DB_A=2.190 | diversite_offre_segments_critere_E | metrics_clustering.json |
| C | Classification | Statut réservation (multi-classes) | RandomForest | Régression logistique | Accuracy / F1 / ROC-AUC sur test (critère C) | Acc=0.3360 / F1=0.3351 / AUC=0.5084 | taux_acceptation_annulation_funnel_critere_C | metrics_classification.json |
| D | Régression | final_price | Ridge | Random Forest | RMSE minimal sur test (CV en amont) | RMSE=1.6094 / MAE=0.7077 / R²=1.000000 | panier_moyen_ca_sum_final_price | metrics_regression.json |
| F | Séries temporelles | nb_fact_rows | Holt_ExponentialSmoothing | ARIMA | RMSE minimal sur holdout | RMSE=38.0473 / MAPE=6.12% / h=3 mois | count_id_reservation_mensuel_anticipation | metrics_timeseries.json |

## Détail par critère (texte)

### Critère E — `metrics_clustering.json`

- **Objectif** : Segmentation des entités pour piloter la diversité par segment (offre / comportement).
- **Cible / construction** : Partition en **k = 3** clusters à partir de features numériques standardisées.
- **Modèle retenu** : **KMeans** (comparé à **AgglomerativeClustering_ward** dans le notebook 01_E).
- **Justification** : Comparaison **KMeans vs clustering hiérarchique (Ward)** : silhouette sur holdout, indice de Davies-Bouldin ; le notebook retient un **modèle principal** et un **second** pour la robustesse.
- **Qualité** : Silhouette holdout ≈ **0.15084658039644186** ; Davies-Bouldin KMeans ≈ **1.8233558099021157**, Agglomerative ≈ **2.1895427944276884** (DB plus bas = clusters plus compacts).
- **KPI** : `diversite_offre_segments_critere_E`

### Critère C — `metrics_classification.json`

- **Objectif** : Prédire le **statut de réservation** (funnel : annulation, confirmation, attente).
- **Cible / construction** : Variable catégorielle : classes `cancelled`, `confirmed`, `pending`.
- **Modèle retenu** : **RandomForest** (comparé à la **régression logistique** dans 02_C).
- **Justification** : Métriques sur **jeu test** (accuracy, F1 pondéré, ROC-AUC) ; le champion minimise l’erreur de généralisation selon la grille critère **C**.
- **Qualité** : Accuracy test ≈ **0.336** ; F1 pondéré ≈ **0.33511945259042036** ; ROC-AUC ≈ **0.5084364054517613**.
- **KPI** : `taux_acceptation_annulation_funnel_critere_C`

### Critère D — `metrics_regression.json`

- **Objectif** : Prédire le **montant** lié à la réservation / panier (KPI rentabilité).
- **Cible / construction** : Variable continue **`final_price`** (features tabulaires du pipeline 03_D).
- **Modèle retenu** : **Ridge** (comparé au **Random Forest**).
- **Justification** : **RMSE minimal sur le test** (avec validation croisée en amont) ; **Ridge** utile si fortes corrélations / besoin de coefficients lisibles.
- **Qualité** : RMSE test ≈ **1.6093767407037995** ; MAE ≈ **0.7077297608776619** ; R² ≈ **0.9999999897610506**.
- **KPI** : `panier_moyen_ca_sum_final_price`

### Critère F — `metrics_timeseries.json`

- **Objectif** : Prévoir l’**évolution mensuelle** d’un agrégat DW (volume d’activité, CA ou panier moyen selon la colonne disponible).
- **Cible / construction** : Série **`nb_fact_rows`** — Volume mensuel d'activité (nombre de lignes de faits comptées dans le DW par mois).
- **Modèle retenu** : **Holt_ExponentialSmoothing** (comparé à **ARIMA** dans 04_F).
- **Justification** : RMSE minimal sur le holdout; si egalite des RMSE, Holt est choisi si rmse_holt <= rmse_arima — holdout **3** mois.
- **Qualité** : RMSE holdout ≈ **38.04731940320474** ; MAPE ≈ **6.124521012382581** % (unité = celle de la série).
- **KPI** : `count_id_reservation_mensuel_anticipation`

## Contenu JSON intégral par fichier

### metrics_classification.json

```json
{
  "task": "classification",
  "criterion": "C",
  "champion_model": "RandomForest",
  "gridsearch_rf_best_params": {
    "clf__max_depth": null,
    "clf__n_estimators": 80
  },
  "gridsearch_lr_best_params": {
    "clf__C": 0.1
  },
  "test_metrics_champion": {
    "accuracy": 0.336,
    "precision_weighted": 0.33505783637998204,
    "recall_weighted": 0.336,
    "f1_weighted": 0.33511945259042036,
    "roc_auc": 0.5084364054517613
  },
  "test_metrics_rf": {
    "accuracy": 0.336,
    "precision_weighted": 0.33505783637998204,
    "recall_weighted": 0.336,
    "f1_weighted": 0.33511945259042036,
    "roc_auc": 0.5084364054517613
  },
  "test_metrics_lr": {
    "accuracy": 0.336,
    "precision_weighted": 0.3349713436777814,
    "recall_weighted": 0.336,
    "f1_weighted": 0.3347418875684245,
    "roc_auc": 0.5150296300910503
  },
  "classes": [
    "cancelled",
    "confirmed",
    "pending"
  ],
  "kpi_alignment": "taux_acceptation_annulation_funnel_critere_C"
}
```

### metrics_clustering.json

```json
{
  "task": "clustering",
  "model_primary": "KMeans",
  "model_secondary": "AgglomerativeClustering_ward",
  "k": 3,
  "silhouette": 0.15084658039644186,
  "silhouette_train": 0.1555073405798967,
  "silhouette_holdout": 0.15084658039644186,
  "silhouette_kmeans_full": 0.15711145953279274,
  "silhouette_agg_full": 0.108250307498888,
  "davies_bouldin_kmeans": 1.8233558099021157,
  "davies_bouldin_agg": 2.1895427944276884,
  "n_samples": 3382,
  "n_train": 2705,
  "n_holdout": 677,
  "kpi_alignment": "diversite_offre_segments_critere_E",
  "cluster_segment_labels_file": "clustering_segment_labels.json",
  "cluster_feature_names_file": "clustering_feature_names.json",
  "cluster_share_train_sample": {
    "0": 0.02365464222353637,
    "1": 0.460969840331165,
    "2": 0.5153755174452986
  }
}
```

### metrics_regression.json

```json
{
  "task": "regression",
  "criterion": "D",
  "champion_model": "Ridge",
  "target": "final_price",
  "kpi_alignment": "panier_moyen_ca_sum_final_price",
  "features": [
    "id_date",
    "id_event",
    "id_servicecategory",
    "id_benchmark",
    "id_provider",
    "service_price",
    "benchmark_avg_price",
    "event_budget",
    "cal_month",
    "cal_year",
    "quarter",
    "commission_margin"
  ],
  "cv_ridge": {
    "model": "Ridge",
    "cv_rmse_mean": 2.0151551596729758,
    "cv_rmse_std": 0.5911394276460822,
    "cv_r2_mean": 0.9999999837267977,
    "cv_mae_mean": 0.8735065911183801
  },
  "cv_random_forest": {
    "model": "RandomForest",
    "cv_rmse_mean": 842.8229287853898,
    "cv_rmse_std": 608.8250143257237,
    "cv_r2_mean": 0.9972010627831338,
    "cv_mae_mean": 128.6542265279889
  },
  "test_ridge": {
    "mse": 2.590093493518385,
    "rmse": 1.6093767407037995,
    "mae": 0.7077297608776619,
    "r2": 0.9999999897610506
  },
  "test_random_forest": {
    "mse": 242238.000446797,
    "rmse": 492.1767979565849,
    "mae": 95.38928216838761,
    "r2": 0.9990424042109068
  },
  "test_champion": {
    "mse": 2.590093493518385,
    "rmse": 1.6093767407037995,
    "mae": 0.7077297608776619,
    "r2": 0.9999999897610506
  }
}
```

### metrics_timeseries.json

```json
{
  "task": "time_series",
  "criterion": "F",
  "series": "nb_fact_rows",
  "champion_model": "Holt_ExponentialSmoothing",
  "champion_rule": "RMSE minimal sur le holdout; si egalite des RMSE, Holt est choisi si rmse_holt <= rmse_arima",
  "target_column_explained": "Volume mensuel d'activité (nombre de lignes de faits comptées dans le DW par mois).",
  "rmse_delta_holt_minus_arima": -0.60296772,
  "adf_pvalue": 0.0002496300832538065,
  "kpss_pvalue": 0.1,
  "decomposition_period_used": 12,
  "test_holt": {
    "rmse": 38.04731940320474,
    "mae": 34.549318718502946,
    "mape": 6.124521012382581
  },
  "test_arima": {
    "rmse": 38.65028712749131,
    "mae": 34.29316623613181,
    "mape": 6.144083281318748
  },
  "test_champion": {
    "rmse": 38.04731940320474,
    "mae": 34.549318718502946,
    "mape": 6.124521012382581
  },
  "horizon": 3,
  "kpi_alignment": "count_id_reservation_mensuel_anticipation"
}
```

## Tableau agrégé (CSV)

Les colonnes diffèrent selon les tâches ; le JSON ci-dessus reste la référence.

```csv
task,criterion,champion_model,gridsearch_rf_best_params,gridsearch_lr_best_params,test_metrics_champion,test_metrics_rf,test_metrics_lr,classes,kpi_alignment,model_primary,model_secondary,k,silhouette,silhouette_train,silhouette_holdout,silhouette_kmeans_full,silhouette_agg_full,davies_bouldin_kmeans,davies_bouldin_agg,n_samples,n_train,n_holdout,cluster_segment_labels_file,cluster_feature_names_file,cluster_share_train_sample,target,features,cv_ridge,cv_random_forest,test_ridge,test_random_forest,test_champion,series,champion_rule,target_column_explained,rmse_delta_holt_minus_arima,adf_pvalue,kpss_pvalue,decomposition_period_used,test_holt,test_arima,horizon
classification,C,RandomForest,"{'clf__max_depth': None, 'clf__n_estimators': 80}",{'clf__C': 0.1},"{'accuracy': 0.336, 'precision_weighted': 0.33505783637998204, 'recall_weighted': 0.336, 'f1_weighted': 0.33511945259042036, 'roc_auc': 0.5084364054517613}","{'accuracy': 0.336, 'precision_weighted': 0.33505783637998204, 'recall_weighted': 0.336, 'f1_weighted': 0.33511945259042036, 'roc_auc': 0.5084364054517613}","{'accuracy': 0.336, 'precision_weighted': 0.3349713436777814, 'recall_weighted': 0.336, 'f1_weighted': 0.3347418875684245, 'roc_auc': 0.5150296300910503}","['cancelled', 'confirmed', 'pending']",taux_acceptation_annulation_funnel_critere_C,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
clustering,,,,,,,,,diversite_offre_segments_critere_E,KMeans,AgglomerativeClustering_ward,3.0,0.15084658039644186,0.1555073405798967,0.15084658039644186,0.15711145953279274,0.108250307498888,1.8233558099021157,2.1895427944276884,3382.0,2705.0,677.0,clustering_segment_labels.json,clustering_feature_names.json,"{'0': 0.02365464222353637, '1': 0.460969840331165, '2': 0.5153755174452986}",,,,,,,,,,,,,,,,,
regression,D,Ridge,,,,,,,panier_moyen_ca_sum_final_price,,,,,,,,,,,,,,,,,final_price,"['id_date', 'id_event', 'id_servicecategory', 'id_benchmark', 'id_provider', 'service_price', 'benchmark_avg_price', 'event_budget', 'cal_month', 'cal_year', 'quarter', 'commission_margin']","{'model': 'Ridge', 'cv_rmse_mean': 2.0151551596729758, 'cv_rmse_std': 0.5911394276460822, 'cv_r2_mean': 0.9999999837267977, 'cv_mae_mean': 0.8735065911183801}","{'model': 'RandomForest', 'cv_rmse_mean': 842.8229287853898, 'cv_rmse_std': 608.8250143257237, 'cv_r2_mean': 0.9972010627831338, 'cv_mae_mean': 128.6542265279889}","{'mse': 2.590093493518385, 'rmse': 1.6093767407037995, 'mae': 0.7077297608776619, 'r2': 0.9999999897610506}","{'mse': 242238.000446797, 'rmse': 492.1767979565849, 'mae': 95.38928216838761, 'r2': 0.9990424042109068}","{'mse': 2.590093493518385, 'rmse': 1.6093767407037995, 'mae': 0.7077297608776619, 'r2': 0.9999999897610506}",,,,,,,,,,
time_series,F,Holt_ExponentialSmoothing,,,,,,,count_id_reservation_mensuel_anticipation,,,,,,,,,,,,,,,,,,,,,,,"{'rmse': 38.04731940320474, 'mae': 34.549318718502946, 'mape': 6.124521012382581}",nb_fact_rows,"RMSE minimal sur le holdout; si egalite des RMSE, Holt est choisi si rmse_holt <= rmse_arima",Volume mensuel d'activité (nombre de lignes de faits comptées dans le DW par mois).,-0.60296772,0.0002496300832538065,0.1,12.0,"{'rmse': 38.04731940320474, 'mae': 34.549318718502946, 'mape': 6.124521012382581}","{'rmse': 38.65028712749131, 'mae': 34.29316623613181, 'mape': 6.144083281318748}",3.0

```
