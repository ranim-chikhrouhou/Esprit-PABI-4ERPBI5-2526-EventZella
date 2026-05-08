# Rapport d'Audit de Performance — Critère F
## Projet EventZella — Power BI + SQL Server

---

## 1. Contexte & Objectif

**Objectif** : Identifier les goulots d'étranglement de performance dans les rapports Power BI
et les requêtes DAX sous-jacentes afin d'optimiser les temps de chargement et d'actualisation.

**Outils utilisés** :
- Power BI Performance Analyzer (intégré à Power BI Desktop)
- Fichier exporté : `PowerBIPerformanceData.json` (1 574 476 bytes)
- Base de données : `DW_eventzella` sur `ASUSRANIM`

**Données capturées** :
- **7 272 événements** enregistrés au total
- **744 visuels** analysés (chargements + interactions)
- **744 requêtes DAX** mesurées
- **744 rendus visuels** mesurés

---

## 2. Statistiques Globales de Performance

| Indicateur | Valeur mesurée | Seuil acceptable | Statut |
|---|---|---|---|
| Visuel le plus lent | **1 037 ms** | < 1 000 ms | ❌ Dépasse |
| Visuel le plus rapide | **24 ms** | — | ✅ Excellent |
| Temps moyen par visuel | **359 ms** | < 500 ms | ✅ Acceptable |
| Requête DAX la plus lente | **860 ms** | < 500 ms | ⚠️ Élevée |
| Moyenne requêtes DAX | **172 ms** | < 300 ms | ✅ Bonne |
| Visuels > 1 000 ms (lents) | **3 visuels** | 0 idéalement | ❌ |
| Visuels > 300 ms (moyens) | **391 visuels** | Minimiser | ⚠️ |
| Visuels ≤ 300 ms (rapides) | **353 visuels** | Maximiser | ✅ |

---

## 3. Top Visuels les Plus Lents

### Visuels critiques (> 1 000 ms) ❌

| Rang | Visuel | Type | Durée | Page concernée |
|---|---|---|---|---|
| 1 | Complaints Resolution Rate | `cardVisual` | **1 037 ms** | Satisfaction Client (CRM) |
| 2 | Total Complaints Received | `cardVisual` | **1 010 ms** | Satisfaction Client (CRM) |
| 3 | Reservations Cancellation Rate | `cardVisual` | **1 006 ms** | Performance Commerciale |

### Visuels lourds (700 ms – 1 000 ms) ⚠️

| Visuel | Type | Durée | Observation |
|---|---|---|---|
| Tital Number of Complaints | `cardVisual` | 993 ms | Mesure COUNT sur Fact_SatisfactionClient |
| Net Promoter Score | `cardVisual` | 975 ms | Calcul NPS complexe (promoteurs - détracteurs) |
| Recurring Beneficiaries | `cardVisual` | 975 ms | Jointure DimBeneficiary + agrégation |
| Average Reservation Frequency | `cardVisual` | 949 ms | Fréquence calculée sur l'historique complet |
| Confirmed Reservations Percentage | `cardVisual` | 949 ms | Ratio conditionnel sans filtre de base |
| Provider Complaints vs. Resolution | `lineStackedColumnComboChart` | 932 ms | Double axe, multi-mesures |
| Complaint Resolution Efficiency | `gauge` | 930 ms | Jauge avec cible dynamique |
| Complaint Volume by Event Type | `stackedAreaChart` | 927 ms | Série temporelle par catégorie |
| Resolution Status Breakdown | `donutChart` | 925 ms | Donut avec 5+ segments |
| Venue Density by Governorate | `azureMap` | 742 ms | Carte Azure Maps (données géographiques) |

---

## 4. Analyse par Type de Visuel

| Type de visuel | Nb occurrences | Durée moyenne | Observations |
|---|---|---|---|
| `cardVisual` | ~120 | ~780 ms | Le plus lent en moyenne — KPIs complexes |
| `lineStackedColumnComboChart` | ~15 | ~700 ms | Double axe coûteux |
| `gauge` | ~10 | ~650 ms | Cible dynamique recalculée |
| `stackedAreaChart` | ~20 | ~620 ms | Séries temporelles volumineuses |
| `donutChart` | ~25 | ~580 ms | Agrégations multiples |
| `clusteredBarChart` | ~30 | ~520 ms | Acceptable |
| `areaChart` | ~20 | ~500 ms | Acceptable |
| `azureMap` | ~5 | ~742 ms | Chargement tuiles + données |
| `slicer` | ~40 | ~400 ms | Impact sur les autres visuels |
| `image` | ~15 | ~200 ms | Rapide |
| `actionButton` | ~20 | ~665 ms | Navigation entre pages |

---

## 5. Analyse des Requêtes DAX

### Requêtes les plus lentes
| Requête (mesure) | Durée DAX | Cause probable |
|---|---|---|
| Complaints Resolution Rate | ~860 ms | DIVIDE + CALCULATE + FILTER imbriqués |
| Net Promoter Score | ~820 ms | Logique promoteurs/détracteurs en DAX pur |
| Reservations Cancellation Rate | ~810 ms | Ratio conditionnel sur grande table de faits |
| Average Reservation Frequency | ~780 ms | Comptage distinct sur fenêtre temporelle |

### Répartition des requêtes DAX
- Requêtes < 100 ms : **~40%** des requêtes ✅
- Requêtes 100–300 ms : **~35%** des requêtes ✅
- Requêtes 300–500 ms : **~15%** des requêtes ⚠️
- Requêtes > 500 ms : **~10%** des requêtes ❌

---

## 6. Goulots d'Étranglement Identifiés

### 6.1 Côté Power BI (DAX & visuels)

| Problème | Impact | Recommandation |
|---|---|---|
| `cardVisual` avec mesures complexes | +1 000 ms | Pré-calculer dans le modèle ou simplifier la formule DAX |
| Mesures NPS et taux sans variables DAX | Recalcul complet à chaque filtre | Utiliser `VAR` pour stocker les résultats intermédiaires |
| `azureMap` avec données géographiques | 742 ms | Limiter le nombre de points affichés, agréger par région |
| Slicers sans filtre de base | Chargement de toutes les valeurs | Ajouter une valeur par défaut sur les slicers de date |
| `actionButton` (665 ms) | Lent pour un bouton | Précharger la page de destination |

### 6.2 Côté SQL Server / Modèle de données

| Problème | Impact | Recommandation |
|---|---|---|
| Tables de faits sans index sur FK | Scans complets à chaque requête | Créer des index non-cluster sur `id_date`, `id_provider` |
| Pas d'agrégations définies | Chaque visuel relit toute la table | Définir des tables d'agrégation Power BI |
| Statistiques SQL non mises à jour | Plans d'exécution sous-optimaux | `UPDATE STATISTICS` hebdomadaire |

---

## 7. Recommandations d'Optimisation SQL

```sql
-- Index pour Fact_PerformanceCommerciale (accélère les KPIs Marketing)
CREATE NONCLUSTERED INDEX IX_Fact_Perf_Date
    ON Fact_PerformanceCommerciale (id_date)
    INCLUDE (final_price, id_reservation);

CREATE NONCLUSTERED INDEX IX_Fact_Perf_Provider
    ON Fact_PerformanceCommerciale (id_provider, id_servicecategory)
    INCLUDE (final_price);

-- Index pour Fact_SatisfactionClient (accélère les KPIs CRM)
CREATE NONCLUSTERED INDEX IX_Fact_Satisf_Complaint
    ON Fact_SatisfactionClient (id_complaint, id_date)
    INCLUDE (id_feedback, id_provider);

-- Index pour Fact_RentabiliteFinanciere (accélère les KPIs Finance)
CREATE NONCLUSTERED INDEX IX_Fact_Fin_Benchmark
    ON Fact_RentabiliteFinanciere (id_benchmark, id_servicecategory)
    INCLUDE (final_price);

-- Mise à jour des statistiques
UPDATE STATISTICS DW_eventzella;
```

---

## 8. Recommandations DAX

```dax
-- AVANT (lent) : NPS calculé directement
Net Promoter Score =
DIVIDE(
    COUNTROWS(FILTER(Fact_SatisfactionClient, Fact_SatisfactionClient[rating] >= 9)) -
    COUNTROWS(FILTER(Fact_SatisfactionClient, Fact_SatisfactionClient[rating] <= 6)),
    COUNTROWS(Fact_SatisfactionClient)
) * 100

-- APRÈS (optimisé) : utilisation de VAR
Net Promoter Score Optimisé =
VAR TotalRepondants = COUNTROWS(Fact_SatisfactionClient)
VAR Promoteurs = CALCULATE(COUNTROWS(Fact_SatisfactionClient),
                           Fact_SatisfactionClient[rating] >= 9)
VAR Detracteurs = CALCULATE(COUNTROWS(Fact_SatisfactionClient),
                            Fact_SatisfactionClient[rating] <= 6)
RETURN
    IF(TotalRepondants = 0, BLANK(),
       DIVIDE(Promoteurs - Detracteurs, TotalRepondants) * 100)
```

---

## 9. Conclusion

### Bilan global
Le rapport EventZella présente des performances **globalement acceptables** avec une moyenne de **359 ms** par visuel.

**Points forts :**
- 47% des visuels chargent en moins de 300 ms
- La requête DAX moyenne (172 ms) est excellente
- Le visuel le plus rapide atteint 24 ms

**Points d'amélioration :**
- 3 visuels dépassent 1 seconde (seuil critique)
- Les `cardVisual` avec logique métier complexe (NPS, taux annulation) sont systématiquement lents
- La carte Azure Maps nécessite une agrégation préalable des données

### Gain estimé avec optimisations
| Optimisation | Gain estimé |
|---|---|
| Index SQL sur FK des facts | -30 à -40% sur les requêtes lentes |
| Variables DAX (`VAR`) | -20 à -30% sur les cardVisuels |
| Table d'agrégation Power BI | -50% sur les visuels > 500ms |
| Filtre par défaut sur slicers | -15% sur le chargement initial |

---

*Rapport généré à partir de `PowerBIPerformanceData.json` — 7 272 événements analysés*
*Outil : Power BI Performance Analyzer — Projet EventZella S11*
