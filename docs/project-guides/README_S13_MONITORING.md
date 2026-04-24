# 🎯 Week S13: Production Monitoring System

## Quick Start (5 Minutes)

### Option 1: Automated Start (Windows)

```bash
# Double-click or run:
start_monitoring.bat
```

This will:
1. Install dependencies
2. Start Docker services (FastAPI, Prometheus, Grafana)
3. Open Grafana and Prometheus in browser
4. Display access information

### Option 2: Manual Start

```bash
# 1. Install dependencies
pip install -r requirements_monitoring.txt

# 2. Start services
docker-compose -f docker-compose-monitoring.yml up -d

# 3. Wait 30 seconds for initialization

# 4. Access services
# Grafana: http://localhost:3000 (admin/eventzilla2026)
# Prometheus: http://localhost:9090
# FastAPI: http://localhost:8000
```

---

## 📊 View Dashboard

1. Open **Grafana**: http://localhost:3000
2. Login: `admin` / `eventzilla2026`
3. Navigate to: **Dashboards** → **EventZilla MLOps - Production Monitoring**

---

## 🎭 Run Simulations

```bash
python simulate_scenarios.py
```

This will run 3 scenarios:
1. **High Traffic** (60s) - 10 req/s
2. **API Errors** (30s) - 30% error rate
3. **Model Drift** (45s) - Data distribution shift

**Watch the dashboard while simulations run!**

---

## 📋 What's Included

### Files Created

```
PI BI NEW (2)/PI BI NEW/
├── ML/api/monitoring.py                    # Monitoring module
├── prometheus.yml                          # Prometheus config
├── prometheus_rules.yml                    # Alert rules
├── docker-compose-monitoring.yml           # Docker services
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml     # Datasource config
│   │   └── dashboards/dashboards.yml      # Dashboard provisioning
│   └── dashboards/
│       └── eventzilla_mlops_dashboard.json # Main dashboard
├── simulate_scenarios.py                   # Simulation script
├── start_monitoring.bat                    # Quick start script
├── MONITORING_GUIDE_S13.md                 # Complete guide
└── README_S13_MONITORING.md                # This file
```

### Metrics Collected

- **Traffic**: Request rate, by model, by status
- **Performance**: Latency (p50, p95, p99)
- **Stability**: Error rate, success rate
- **Model Health**: Accuracy, confidence, predictions count
- **Data Quality**: Missing values, freshness, drift detection
- **System**: CPU, memory, disk usage

### Alerts Configured

- High latency (>1s warning, >5s critical)
- High error rate (>5% warning, >20% critical)
- Model accuracy drop (<0.80 warning, <0.70 critical)
- Low confidence (<0.60)
- Data drift detected
- Model drift detected
- High missing values (>15%)
- Stale data (>1 hour)
- System resources (CPU >80%, Memory >85%, Disk >90%)

---

## 🎯 Deliverables Checklist

- [x] **Prometheus monitoring working** ✅
  - Scraping every 10 seconds
  - Metrics endpoint `/metrics`
  - Real-time activity tracking

- [x] **Grafana dashboard** ✅
  - Traffic, performance, stability panels
  - Model health (accuracy, confidence)
  - Data health (missing values, freshness)
  - Clear and interpretable
  - Storytelling approach

- [x] **Drift & degradation detection** ✅
  - Data distribution shift
  - Accuracy drop (>5%)
  - Confidence decrease
  - Threshold-based rules

- [x] **Alerting system** ✅
  - High latency alerts
  - High error rate alerts
  - Accuracy degradation alerts
  - Drift detection alerts
  - Logs and notifications

- [x] **Simulation scenarios** ✅
  - High traffic → latency impact
  - API errors → error spike
  - Model drift → performance degradation
  - Automated script

- [x] **Observability** ✅
  - Metrics (what happens)
  - Logs (why it happens)
  - Errors, anomalies, retraining triggers

- [x] **Baseline comparison** ✅
  - Baseline values defined
  - Deviations detectable
  - Alerts on threshold breach

---

## 📸 Screenshots for Deliverables

### 1. Grafana Dashboard
- Full dashboard view
- All panels visible
- During/after simulation

### 2. Prometheus Alerts
- URL: http://localhost:9090/alerts
- Active alerts
- Firing alerts during simulation

### 3. Metrics Sample
- URL: http://localhost:8000/metrics
- Raw Prometheus metrics

---

## 🔧 Troubleshooting

### Services Not Starting

```bash
# Check Docker is running
docker ps

# View logs
docker-compose -f docker-compose-monitoring.yml logs

# Restart services
docker-compose -f docker-compose-monitoring.yml down
docker-compose -f docker-compose-monitoring.yml up -d
```

### Dashboard Shows "No Data"

1. Verify Prometheus is scraping: http://localhost:9090/targets
2. Check FastAPI metrics: http://localhost:8000/metrics
3. Generate traffic: `python simulate_scenarios.py`
4. Refresh Grafana dashboard (Ctrl+R)

### Alerts Not Firing

1. Check alert rules: http://localhost:9090/rules
2. Verify metrics are collected
3. Run simulations to trigger alerts
4. Check `for` duration in rules (alerts need sustained condition)

---

## 📚 Documentation

- **Complete Guide**: `MONITORING_GUIDE_S13.md`
- **Monitoring Module**: `ML/api/monitoring.py`
- **Dashboard JSON**: `grafana/dashboards/eventzilla_mlops_dashboard.json`

---

## 🎉 Quick Demo Script

```bash
# 1. Start services
start_monitoring.bat

# 2. Open Grafana
# http://localhost:3000 (admin/eventzilla2026)

# 3. Open dashboard
# "EventZilla MLOps - Production Monitoring"

# 4. Run simulations (in new terminal)
python simulate_scenarios.py

# 5. Watch dashboard update in real-time!
# - Traffic increases
# - Errors spike
# - Drift detected

# 6. Check alerts
# http://localhost:9090/alerts

# 7. Take screenshots for deliverables
```

---

## ✅ Summary

**All Week S13 requirements implemented:**

1. ✅ Prometheus monitoring (10s scrape interval)
2. ✅ Grafana dashboard (traffic, performance, model health, data quality)
3. ✅ Drift detection (data & model)
4. ✅ Alerting system (multi-level, multi-category)
5. ✅ Simulation scenarios (high traffic, errors, drift)
6. ✅ Observability (metrics + logs)
7. ✅ Baseline comparison (accuracy, latency, confidence)

**Ready for demonstration and evaluation!** 🚀

---

**Need Help?**
- Read: `MONITORING_GUIDE_S13.md` (complete guide)
- Check: Troubleshooting section above
- Verify: All services running with `docker ps`

---

**Last Updated**: 2026-05-03  
**Version**: 1.0
