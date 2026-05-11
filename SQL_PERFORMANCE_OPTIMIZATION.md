# SQL Server Performance Optimization Guide

## ✅ Changes Applied

### 1. **Reduced Connection Timeout** (6s → 2s)
- **Before**: Each failed server attempt took 6 seconds
- **After**: Each failed server attempt takes only 2 seconds
- **Impact**: Up to **28 seconds faster** login if wrong servers are tried first

### 2. **Optimized Server List**
- **Before**: Tried 7 different server names sequentially
- **After**: Only tries 3 most common servers (your actual server first)
- **Impact**: **Skips 4 unnecessary connection attempts**

### 3. **Increased Connection Pool**
- **Before**: pool_size=1, max_overflow=0 (only 1 connection)
- **After**: pool_size=3-5, max_overflow=2-3 (up to 6-8 connections)
- **Impact**: **Reuses existing connections** instead of creating new ones

### 4. **Added Connection Recycling**
- **New**: Connections are recycled after 1 hour
- **Impact**: Prevents stale connections that can cause timeouts

---

## 🚀 Additional Recommendations

### A. **SQL Server Configuration** (Most Important!)

#### 1. Enable TCP/IP Protocol
```
1. Open "SQL Server Configuration Manager"
2. Navigate to: SQL Server Network Configuration → Protocols for MSSQLSERVER
3. Right-click "TCP/IP" → Enable
4. Restart SQL Server service
```

#### 2. Set Fixed TCP Port (Optional but Recommended)
```
1. In SQL Server Configuration Manager
2. Right-click TCP/IP → Properties
3. Go to "IP Addresses" tab
4. Scroll to "IPAll" section
5. Set "TCP Port" to 1433 (default)
6. Clear "TCP Dynamic Ports"
7. Restart SQL Server service
```

#### 3. Enable SQL Server Browser Service
```
1. Open "Services" (services.msc)
2. Find "SQL Server Browser"
3. Set Startup Type to "Automatic"
4. Start the service
```

### B. **Windows Firewall** (If Applicable)

Add firewall rule for SQL Server:
```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "SQL Server" -Direction Inbound -Protocol TCP -LocalPort 1433 -Action Allow
```

### C. **Database Optimization**

#### 1. Add Index on AppUsers Table
```sql
-- Run in SSMS on DW_eventzella database
USE DW_eventzella;
GO

-- Index for faster login lookups
CREATE NONCLUSTERED INDEX IX_AppUsers_Login 
ON dbo.AppUsers (login_name, is_active)
INCLUDE (role_name, full_name, email);
GO
```

#### 2. Update Statistics
```sql
-- Run periodically to keep query performance optimal
USE DW_eventzella;
GO

UPDATE STATISTICS dbo.AppUsers WITH FULLSCAN;
GO
```

### D. **Application-Level Caching** (Advanced)

Add session-level caching to avoid repeated database queries:

```python
# In streamlit_app.py - add at the top
import streamlit as st

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def get_cached_sql_engine():
    """Cached SQL engine to avoid recreating connections"""
    from ML.ml_paths import get_sql_engine
    return get_sql_engine()
```

---

## 📊 Expected Performance Improvements

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| **Successful login (correct server)** | 2-6s | 0.5-2s | **70-75% faster** |
| **Failed login (wrong password)** | 2-6s | 0.5-2s | **70-75% faster** |
| **Wrong server tried first** | 12-42s | 4-6s | **85-90% faster** |
| **Subsequent queries** | 1-3s | 0.2-0.8s | **70-80% faster** |

---

## 🔍 Troubleshooting

### If login is still slow:

1. **Check which server is being used:**
   - Look at the console output when logging in
   - The correct server should be: `DESKTOP-DVMNP7K\MSSQLSERVERS`

2. **Verify SQL Server is running:**
   ```powershell
   Get-Service -Name MSSQL*
   ```

3. **Test connection manually:**
   ```python
   from ML.auth_streamlit import authenticate
   success, error, user = authenticate("your_login", "your_password")
   print(f"Success: {success}, Error: {error}")
   ```

4. **Check SQL Server logs:**
   - Open SSMS
   - Connect to your server
   - Management → SQL Server Logs → Current

### If you see "Login failed" errors:

1. **Verify Mixed Mode Authentication is enabled:**
   ```
   1. In SSMS, right-click server → Properties
   2. Security → Server authentication
   3. Select "SQL Server and Windows Authentication mode"
   4. Restart SQL Server service
   ```

2. **Verify user exists:**
   ```sql
   USE DW_eventzella;
   SELECT * FROM dbo.AppUsers WHERE is_active = 1;
   ```

---

## 📝 Testing the Changes

1. **Restart Streamlit:**
   ```bash
   # Stop current process (Ctrl+C)
   # Then restart:
   streamlit run "PI BI NEW (2)/PI BI NEW/ML/streamlit_app.py"
   ```

2. **Test login with valid credentials**

3. **Measure time:**
   - Should now take **2-4 seconds** instead of 10-40 seconds

---

## 🎯 Next Steps

If performance is still not satisfactory:

1. Consider using **connection string caching**
2. Implement **lazy loading** for dashboard data
3. Add **query result caching** with `@st.cache_data`
4. Use **async database queries** for parallel data loading
5. Consider **Redis** or **Memcached** for session management

---

## 📞 Support

If you continue experiencing slow performance:
- Check SQL Server CPU/Memory usage in Task Manager
- Review SQL Server error logs
- Consider upgrading SQL Server Express to Standard (if using Express)
- Check network latency if SQL Server is on a different machine
