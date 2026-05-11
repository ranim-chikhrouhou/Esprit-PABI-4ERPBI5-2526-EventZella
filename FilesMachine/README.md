# FilesMachine — données projet (SA, DW, CSV)

## Sauvegardes SQL Server (`DB/`)

Fichiers présents dans le dépôt (noms réels) :

| Fichier | Rôle |
|---------|------|
| **`DW_Eventzilla`** | **Data Warehouse** — entrepôt en étoile (faits + dimensions **nettoyés**). **Base à utiliser pour le ML et les notebooks `ML/notebooks/*.ipynb`**. |
| **`SA_eventzilla`** | **Staging Area** — zone de chargement amont (moins adaptée comme source analytique finale). |

Restauration typique (adapter chemins et noms de fichiers logiques dans le `.bak`) :

```sql
RESTORE DATABASE [DW_Eventzilla]
FROM DISK = N'C:\chemin\vers\PI BI NEW\FilesMachine\DB\DW_Eventzilla'
WITH REPLACE, RECOVERY;

RESTORE DATABASE [SA_eventzilla]
FROM DISK = N'C:\chemin\vers\PI BI NEW\FilesMachine\DB\SA_eventzilla'
WITH REPLACE, RECOVERY;
```

> Utilisez **FILELISTONLY** / **HEADERONLY** sur le fichier de backup si les noms logiques diffèrent.

Connexion Python : variable d’environnement **`EVENTZILLA_SQL_URI`** pointant vers **`DW_Eventzilla`** (voir `ML/README.md`).

## Schéma DW (référence projet)

Tables et clés : **`ScriptsDiagrams/EventZilla_DWH_Model.dbml`**, **`EventZilla_DWH_Par_Fact_Mermaid.md`**.  
Noms **Power BI** utilisés dans les requêtes ML : **`ML/schema_eventzilla.py`** (`Fact_RentabiliteFinanciere`, `Fact_PerformanceCommerciale`, `DimReservation`, `DimDate`, …).

## Autres dossiers

- **`data_original/`** — CSV sources initiaux (fallback si pas de SQL).
- **`datascrapped/`** — données externes (benchmark, jours fériés, tendances, venues…).
- **`EventZilla_Dashboards_Improved.pdf`** — référence visuels / parties données manquantes.
- Liste KPI anglais : `docs/Liste_Des_Kpis_Updated_English_DAX.md` ou copie PDF sous `FilesMachine/`.
