-- ================================================================
-- EventZilla — Performance Optimization for AppUsers Login
-- Run this script to speed up login queries
-- ================================================================

USE DW_eventzella;
GO

PRINT '════════════════════════════════════════════════════════════';
PRINT 'EventZilla - AppUsers Performance Optimization';
PRINT '════════════════════════════════════════════════════════════';
PRINT '';

-- ────────────────────────────────────────────────────────────────
-- STEP 1: Add optimized index for login queries
-- ────────────────────────────────────────────────────────────────

-- Check if index already exists
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_AppUsers_Login_Active' 
    AND object_id = OBJECT_ID('dbo.AppUsers')
)
BEGIN
    PRINT 'Creating optimized index on AppUsers...';
    
    -- Create covering index for the login query
    -- INCLUDE clause adds columns to the index leaf level
    -- This allows the query to be satisfied entirely from the index
    CREATE NONCLUSTERED INDEX IX_AppUsers_Login_Active
    ON dbo.AppUsers (login_name, is_active)
    INCLUDE (role_name, full_name, email)
    WITH (FILLFACTOR = 90, PAD_INDEX = ON);
    
    PRINT '✓ Index IX_AppUsers_Login_Active created successfully';
END
ELSE
BEGIN
    PRINT '✓ Index IX_AppUsers_Login_Active already exists';
END
GO

-- ────────────────────────────────────────────────────────────────
-- STEP 2: Update statistics for optimal query plans
-- ────────────────────────────────────────────────────────────────

PRINT '';
PRINT 'Updating statistics on AppUsers table...';

UPDATE STATISTICS dbo.AppUsers WITH FULLSCAN;

PRINT '✓ Statistics updated successfully';
GO

-- ────────────────────────────────────────────────────────────────
-- STEP 3: Test query performance
-- ────────────────────────────────────────────────────────────────

PRINT '';
PRINT 'Testing query performance...';
PRINT '';

-- Enable execution time display
SET STATISTICS TIME ON;
SET STATISTICS IO ON;

-- Test the actual login query
SELECT login_name, role_name, full_name, email
FROM   dbo.AppUsers
WHERE  login_name = 'ranim_chikhrouhou'
  AND  is_active  = 1;

SET STATISTICS TIME OFF;
SET STATISTICS IO OFF;

PRINT '';
PRINT '════════════════════════════════════════════════════════════';
PRINT 'Optimization Complete!';
PRINT '';
PRINT 'Expected improvements:';
PRINT '  - Login query: 50-80% faster';
PRINT '  - Index seeks instead of table scans';
PRINT '  - Reduced I/O operations';
PRINT '';
PRINT 'Next steps:';
PRINT '  1. Restart your Streamlit app';
PRINT '  2. Test login - should be much faster now';
PRINT '════════════════════════════════════════════════════════════';
GO
