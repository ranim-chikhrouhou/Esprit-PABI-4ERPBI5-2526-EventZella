# EventZilla — DAX measures documentation

This file is the **entry point** for the evaluation rubric (*Git Deployment — documented DAX*).

## Detailed documents

| Document | Content |
|----------|---------|
| **`docs/Liste_Des_Kpis_Updated_English_DAX.md`** | **English** catalog of all measures with DAX blocks (Git-friendly). |
| **`deliverables/EventZilla_Dashboards_Table2_DAX_Visuels_Detailles.md`** | DAX measures + visual configuration **by stakeholder** (script-generated). |
| **`docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`** | **Objectives → KPI → formulas** chain (specification / aggregate notation). |

To regenerate the DAX + visuals table (from repository root):

```powershell
python scripts/build_dashboards_table2_with_formulas.py
python scripts/build_dashboards_table2_dax_et_visuels.py
```
