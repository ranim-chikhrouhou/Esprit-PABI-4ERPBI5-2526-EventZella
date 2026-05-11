# All UI Files in EventZilla Project

## 🎨 Main UI Applications

### 1. **Streamlit Web Application** (Primary UI)
- **Main file**: `ML/streamlit_app.py` ⭐ **MAIN UI FILE**
- **Backup**: `ML/streamlit_app.py.french_backup`
- **Authentication**: `ML/auth_streamlit.py`
- **Configuration**: `.streamlit/config.toml`
- **Status**: ✅ Fully translated to English
- **URL**: http://localhost:8502

### 2. **FastAPI Backend API**
- **Main file**: `ML/api/main.py`
- **Authentication**: `ML/api/auth_sql.py`
- **Init**: `ML/api/__init__.py`
- **Launcher**: `start_api.bat`
- **Status**: Backend API (JSON responses)
- **URL**: http://localhost:8000

### 3. **n8n Workflow Automation** (Visual Workflow UI)
- **Marketing workflow**: `n8n/workflow_marketing.json`
- **Finance workflow**: `n8n/workflow_finance.json`
- **CRM workflow**: `n8n/workflow_crm.json`
- **Error handler**: `n8n/workflow_error_handler.json`
- **Test script**: `n8n/test_workflows.py`
- **Status**: JSON workflow definitions
- **URL**: http://localhost:5678

## 📊 Dashboard & Visualization Files

### 4. **HTML Dashboards** (Static Reports)
- `deliverables/EventZilla_Dashboards_KPIs_Objectifs.html`
- `deliverables/EventZilla_Dashboards_Table2_DAX_Visuels_Detailles.html`
- `deliverables/EventZilla_Dashboards_Table2_Avec_Formules_Sans_Predictif.html`

### 5. **Power BI Reports**
- **Main report**: `PABIRanimForNaima.pbix`
- **Backup**: `gitMachine/Reports/PABIRanimForNaima.pbix`

### 6. **Dashboard Mockups** (Design References)
- `Mockups/dashboard_marketing (1).png`
- `Mockups/dashboard_finance (1).png`
- `Mockups/dashboard_client (1).png`
- `DashboardsMockups/dashboard_marketing (1).png`
- `DashboardsMockups/dashboard_finance (1).png`
- `DashboardsMockups/dashboard_client (1).png`

## 🛠️ UI Generation Scripts

### 7. **Dashboard Builders**
- `scripts/build_eventzilla_dashboards_html.py` - Generates HTML dashboards
- `scripts/build_dashboards_table2_with_formulas.py` - Builds formula tables
- `scripts/build_dashboards_table2_dax_et_visuels.py` - Builds DAX visualizations

## 📝 UI-Related Configuration Files

### 8. **Streamlit Configuration**
- `.streamlit/config.toml` - Streamlit theme and settings
- `.streamlit.zip` - Streamlit config archive

### 9. **Launch Scripts**
- `LANCER_PROJET.bat` - Main project launcher
- `start_api.bat` - FastAPI launcher

## 📂 UI File Categories

### **Interactive Web UIs** (User-facing)
1. ✅ **Streamlit App** - `ML/streamlit_app.py` (MAIN)
2. ✅ **n8n Workflows** - Visual workflow editor at http://localhost:5678
3. ✅ **FastAPI Docs** - Auto-generated at http://localhost:8000/docs

### **Static Reports** (Read-only)
4. HTML Dashboards (3 files)
5. Power BI Report (1 file)

### **Design Assets**
6. Dashboard mockups (6 PNG files)

## 🎯 Files That Need Translation

### ✅ Already Translated
- `ML/streamlit_app.py` - **FULLY TRANSLATED TO ENGLISH** ✅

### 🔍 May Need Translation Check
1. **`ML/auth_streamlit.py`** - Authentication UI messages
2. **`ML/api/main.py`** - API endpoint descriptions
3. **HTML Dashboards** (3 files) - May contain French text
4. **n8n Workflows** (4 JSON files) - May contain French node names/descriptions
5. **Dashboard builder scripts** (3 files) - May generate French text

### ❌ No Translation Needed
- Power BI files (.pbix) - Binary format
- Configuration files (.toml)
- Mockup images (.png)
- Launch scripts (.bat)

## 📊 Summary

| Category | Count | Status |
|----------|-------|--------|
| **Main UI Files** | 3 | 1 translated ✅ |
| **HTML Dashboards** | 3 | Need check 🔍 |
| **n8n Workflows** | 4 | Need check 🔍 |
| **Power BI Reports** | 1 | Binary ❌ |
| **Mockups** | 6 | Images ❌ |
| **Scripts** | 3 | Need check 🔍 |
| **Config Files** | 2 | No text ❌ |
| **TOTAL** | 22 | - |

## 🎯 Priority Translation List

If you want to translate ALL UI text, check these files in order:

1. ✅ **`ML/streamlit_app.py`** - DONE
2. 🔍 **`ML/auth_streamlit.py`** - Authentication messages
3. 🔍 **`ML/api/main.py`** - API descriptions
4. 🔍 **`deliverables/*.html`** - HTML dashboards (3 files)
5. 🔍 **`n8n/workflow_*.json`** - Workflow descriptions (4 files)
6. 🔍 **`scripts/build_*.py`** - Dashboard generators (3 files)

## 📌 Notes

- **Main user-facing UI**: Streamlit app (already translated ✅)
- **Secondary UIs**: n8n workflows, FastAPI docs, HTML dashboards
- **Most important**: Streamlit app is the primary interface users see
- **HTML dashboards**: Static reports, may contain French labels
- **n8n workflows**: Visual editor, may have French node names

---

**Last Updated**: May 1, 2026  
**Main UI Status**: ✅ Fully translated to English
