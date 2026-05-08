-- ================================================================
-- EventZilla — Gestion des Logins, Utilisateurs et Rôles
-- Serveur  : ASUSRANIM
-- Base DW  : DW_eventzella
-- Équipe   : Ranim Chikhrouhou | Naïma Sarraj | Anas Allam
-- À exécuter en tant qu'administrateur SQL Server (sa ou Windows Auth)
-- ================================================================
--
-- CORRESPONDANCE RÔLES ↔ MEMBRES DE L'ÉQUIPE
-- ┌──────────────────────────────┬───────────────────────────────┬──────────────────────────────────┐
-- │ Login SQL Server             │ Membre                        │ Rôle décideur                    │
-- ├──────────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
-- │ ranim_chikhrouhou            │ Ranim Chikhrouhou             │ Responsable Marketing            │
-- │ naima_sarraj                 │ Naïma Sarraj                  │ Responsable Financière           │
-- │ anas_allam                   │ Anas Allam                    │ Responsable Relation Client      │
-- └──────────────────────────────┴───────────────────────────────┴──────────────────────────────────┘
--
-- CORRESPONDANCE RÔLES ↔ FACT TABLES (déduite des Foreign Keys)
-- ┌─────────────────────────┬─────────────────────────────────────────────────────────────────────┐
-- │ role_marketing           │ Fact_PerformanceCommerciale + dimensions liées                     │
-- │ role_finance             │ Fact_RentabiliteFinanciere  + dimensions liées                     │
-- │ role_crm                 │ Fact_SatisfactionClient     + dimensions liées                     │
-- └─────────────────────────┴─────────────────────────────────────────────────────────────────────┘
-- ================================================================


-- ────────────────────────────────────────────────────────────────
-- PARTIE 1 : LOGINS AU NIVEAU SERVEUR
-- Basculer sur master avant d'exécuter cette partie
-- ────────────────────────────────────────────────────────────────
USE master;
GO

-- Ranim Chikhrouhou — Responsable Marketing
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = 'ranim_chikhrouhou')
    CREATE LOGIN ranim_chikhrouhou
        WITH PASSWORD        = 'Ranim@Marketing2025!',
             CHECK_POLICY    = ON,
             CHECK_EXPIRATION = OFF;
GO

-- Naïma Sarraj — Responsable Financière
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = 'naima_sarraj')
    CREATE LOGIN naima_sarraj
        WITH PASSWORD        = 'Naima@Finance2025!',
             CHECK_POLICY    = ON,
             CHECK_EXPIRATION = OFF;
GO

-- Anas Allam — Responsable Relation Client
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = 'anas_allam')
    CREATE LOGIN anas_allam
        WITH PASSWORD        = 'Anas@CRM2025!',
             CHECK_POLICY    = ON,
             CHECK_EXPIRATION = OFF;
GO

PRINT '✔ PARTIE 1 : Logins serveur créés avec succès.';
GO


-- ────────────────────────────────────────────────────────────────
-- PARTIE 2 : UTILISATEURS DB + RÔLES dans DW_eventzella
-- ────────────────────────────────────────────────────────────────
USE DW_eventzella;
GO

-- ── Créer les utilisateurs DB liés aux logins serveur ───────────
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'ranim_chikhrouhou')
    CREATE USER ranim_chikhrouhou FOR LOGIN ranim_chikhrouhou;
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'naima_sarraj')
    CREATE USER naima_sarraj FOR LOGIN naima_sarraj;
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'anas_allam')
    CREATE USER anas_allam FOR LOGIN anas_allam;
GO

-- ── Créer les rôles applicatifs ──────────────────────────────────
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'role_marketing' AND type = 'R')
    CREATE ROLE role_marketing;
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'role_finance' AND type = 'R')
    CREATE ROLE role_finance;
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'role_crm' AND type = 'R')
    CREATE ROLE role_crm;
GO

-- ── Assigner chaque utilisateur à son rôle ──────────────────────
ALTER ROLE role_marketing ADD MEMBER ranim_chikhrouhou;
ALTER ROLE role_finance   ADD MEMBER naima_sarraj;
ALTER ROLE role_crm       ADD MEMBER anas_allam;
GO

PRINT '✔ PARTIE 2 : Utilisateurs DB et rôles créés et assignés.';
GO


-- ────────────────────────────────────────────────────────────────
-- PARTIE 3 : TABLE AppUsers
-- Lue par FastAPI pour récupérer le rôle JWT après authentification SQL
-- ────────────────────────────────────────────────────────────────
USE DW_eventzella;
GO

IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'AppUsers'
)
BEGIN
    CREATE TABLE dbo.AppUsers (
        login_name   NVARCHAR(100) NOT NULL PRIMARY KEY,   -- = nom du login SQL Server
        role_name    NVARCHAR(50)  NOT NULL,               -- lu par FastAPI pour le JWT
        full_name    NVARCHAR(200) NOT NULL,
        email        NVARCHAR(200) NOT NULL,
        is_active    BIT           NOT NULL DEFAULT 1,
        created_at   DATETIME      NOT NULL DEFAULT GETDATE()
    );
    PRINT '  → Table AppUsers créée.';
END
ELSE
    PRINT '  → Table AppUsers existe déjà — mise à jour des données.';
GO

-- Insérer ou mettre à jour les 3 membres réels de l'équipe
MERGE dbo.AppUsers AS target
USING (
    VALUES
        ('ranim_chikhrouhou', 'marketing_manager', 'Ranim Chikhrouhou', 'ranim.chikhrouhou@esprit.tn', 1),
        ('naima_sarraj',      'financial_manager', 'Naïma Sarraj',      'naima.sarraj@esprit.tn',      1),
        ('anas_allam',        'crm_manager',       'Anas Allam',        'anas.allam@esprit.tn',         1)
) AS source (login_name, role_name, full_name, email, is_active)
ON target.login_name = source.login_name
WHEN MATCHED THEN
    UPDATE SET
        role_name = source.role_name,
        full_name = source.full_name,
        email     = source.email,
        is_active = source.is_active
WHEN NOT MATCHED THEN
    INSERT (login_name, role_name, full_name, email, is_active)
    VALUES (source.login_name, source.role_name, source.full_name, source.email, source.is_active);
GO

-- Lecture de AppUsers autorisée pour tous (FastAPI en a besoin après authentification)
GRANT SELECT ON dbo.AppUsers TO role_marketing;
GRANT SELECT ON dbo.AppUsers TO role_finance;
GRANT SELECT ON dbo.AppUsers TO role_crm;
GO

PRINT '✔ PARTIE 3 : Table AppUsers configurée avec les 3 membres de l équipe.';
GO


-- ────────────────────────────────────────────────────────────────
-- PARTIE 4A : CRÉATION DES VUES ML
-- SQL_ML_PERFORMANCE_WIDE était une variable Python dans schema_eventzilla.py.
-- On la matérialise ici comme une vraie VIEW SQL Server afin de pouvoir
-- lui accorder des permissions par rôle.
-- ────────────────────────────────────────────────────────────────
USE DW_eventzella;
GO

-- Vue ML principale : Fact_PerformanceCommerciale + DimReservation + DimDate
-- Miroir exact de la chaîne SQL_ML_PERFORMANCE_WIDE dans ML/schema_eventzilla.py
IF EXISTS (SELECT 1 FROM sys.views WHERE name = 'SQL_ML_PERFORMANCE_WIDE' AND schema_id = SCHEMA_ID('dbo'))
    DROP VIEW dbo.SQL_ML_PERFORMANCE_WIDE;
GO

CREATE VIEW dbo.SQL_ML_PERFORMANCE_WIDE AS
SELECT TOP 200000
    f.id_date,
    f.id_event,
    f.id_reservation,
    f.id_beneficiary,
    f.id_servicecategory,
    f.id_provider,
    f.id_visitors,
    f.nb_visitors,
    f.nb_reservations_site,
    f.final_price,
    f.event_budget,
    f.service_price,
    r.status          AS reservation_status,
    d.full_date,
    d.[month]         AS cal_month,
    d.[year]          AS cal_year,
    d.is_holiday
FROM  dbo.Fact_PerformanceCommerciale f
INNER JOIN dbo.DimReservation r ON f.id_reservation = r.id_reservation_SK
INNER JOIN dbo.DimDate        d ON f.id_date         = d.id_date_SK;
GO

-- Vue ML financière : Fact_RentabiliteFinanciere + DimDate
-- Miroir de SQL_ML_FINANCIAL_WIDE dans ML/schema_eventzilla.py
IF EXISTS (SELECT 1 FROM sys.views WHERE name = 'SQL_ML_FINANCIAL_WIDE' AND schema_id = SCHEMA_ID('dbo'))
    DROP VIEW dbo.SQL_ML_FINANCIAL_WIDE;
GO

CREATE VIEW dbo.SQL_ML_FINANCIAL_WIDE AS
SELECT TOP 200000
    f.id_date,
    f.id_event,
    f.id_servicecategory,
    f.id_benchmark,
    f.id_provider,
    f.final_price,
    f.service_price,
    f.benchmark_avg_price,
    f.event_budget,
    d.full_date,
    d.[month]    AS cal_month,
    d.[year]     AS cal_year,
    d.is_holiday
FROM  dbo.Fact_RentabiliteFinanciere f
INNER JOIN dbo.DimDate d ON f.id_date = d.id_date_SK;
GO

PRINT '✔ PARTIE 4A : Vues ML créées (SQL_ML_PERFORMANCE_WIDE, SQL_ML_FINANCIAL_WIDE).';
GO


-- ────────────────────────────────────────────────────────────────
-- PARTIE 4B : PERMISSIONS PAR RÔLE
-- Basées sur les Foreign Keys déclarées pour chaque Fact Table
-- ────────────────────────────────────────────────────────────────
USE DW_eventzella;
GO

-- ════════════════════════════════════════════════════════════════
-- role_marketing → Fact_PerformanceCommerciale
-- FK déclarées : DimDate, DimEvent, DimReservation, DimBeneficiary,
--                DimServiceCategory, DimVisitors, DimProvider
-- Vue ML : SQL_ML_PERFORMANCE_WIDE (segmentation + classification)
-- ════════════════════════════════════════════════════════════════

GRANT SELECT ON dbo.Fact_PerformanceCommerciale  TO role_marketing;
GRANT SELECT ON dbo.DimDate                      TO role_marketing;
GRANT SELECT ON dbo.DimEvent                     TO role_marketing;
GRANT SELECT ON dbo.DimReservation               TO role_marketing;
GRANT SELECT ON dbo.DimBeneficiary               TO role_marketing;
GRANT SELECT ON dbo.DimServiceCategory           TO role_marketing;
GRANT SELECT ON dbo.DimVisitors                  TO role_marketing;
GRANT SELECT ON dbo.DimProvider                  TO role_marketing;
GRANT SELECT ON dbo.SQL_ML_PERFORMANCE_WIDE      TO role_marketing;
GO


-- ════════════════════════════════════════════════════════════════
-- role_finance → Fact_RentabiliteFinanciere
-- FK déclarées : DimDate, DimEvent, DimServiceCategory,
--                DimBenchmarkPrice, DimProvider
-- Vue ML : SQL_ML_FINANCIAL_WIDE (régression montants + séries CA)
-- ════════════════════════════════════════════════════════════════

GRANT SELECT ON dbo.Fact_RentabiliteFinanciere   TO role_finance;
GRANT SELECT ON dbo.DimDate                      TO role_finance;
GRANT SELECT ON dbo.DimEvent                     TO role_finance;
GRANT SELECT ON dbo.DimServiceCategory           TO role_finance;
GRANT SELECT ON dbo.DimBenchmarkPrice            TO role_finance;
GRANT SELECT ON dbo.DimProvider                  TO role_finance;
GRANT SELECT ON dbo.SQL_ML_FINANCIAL_WIDE        TO role_finance;
GO


-- ════════════════════════════════════════════════════════════════
-- role_crm → Fact_SatisfactionClient
-- FK déclarées : DimDate, DimProvider, DimServiceCategory,
--                DimReservation, DimFeedback, DimComplaint
-- Vue ML : SQL_ML_PERFORMANCE_WIDE (anticipation annulations)
-- ════════════════════════════════════════════════════════════════

GRANT SELECT ON dbo.Fact_SatisfactionClient      TO role_crm;
GRANT SELECT ON dbo.DimDate                      TO role_crm;
GRANT SELECT ON dbo.DimProvider                  TO role_crm;
GRANT SELECT ON dbo.DimServiceCategory           TO role_crm;
GRANT SELECT ON dbo.DimReservation               TO role_crm;
GRANT SELECT ON dbo.SQL_ML_PERFORMANCE_WIDE      TO role_crm;

-- DimFeedback et DimComplaint : accordées seulement si les tables existent
IF OBJECT_ID('dbo.DimFeedback',  'U') IS NOT NULL
    EXEC('GRANT SELECT ON dbo.DimFeedback  TO role_crm');
ELSE
    PRINT '  ⚠ DimFeedback introuvable — GRANT ignoré (à accorder manuellement après création).';

IF OBJECT_ID('dbo.DimComplaint', 'U') IS NOT NULL
    EXEC('GRANT SELECT ON dbo.DimComplaint TO role_crm');
ELSE
    PRINT '  ⚠ DimComplaint introuvable — GRANT ignoré (à accorder manuellement après création).';
GO

PRINT '✔ PARTIE 4B : Permissions accordées par rôle selon les Foreign Keys.';
GO


-- ────────────────────────────────────────────────────────────────
-- PARTIE 5 : VÉRIFICATIONS FINALES
-- ────────────────────────────────────────────────────────────────
USE DW_eventzella;
GO

PRINT '';
PRINT '════ VÉRIFICATION 1 : Logins serveur ════';
SELECT
    name        AS login_sql,
    type_desc   AS type_authentification,
    is_disabled AS desactive
FROM sys.server_principals
WHERE name IN ('ranim_chikhrouhou', 'naima_sarraj', 'anas_allam')
ORDER BY name;
GO

PRINT '';
PRINT '════ VÉRIFICATION 2 : Utilisateurs DB et rôles assignés ════';
SELECT
    u.name  AS utilisateur_db,
    r.name  AS role_assigne
FROM sys.database_role_members  drm
JOIN sys.database_principals    r ON r.principal_id = drm.role_principal_id
JOIN sys.database_principals    u ON u.principal_id = drm.member_principal_id
WHERE r.name IN ('role_marketing', 'role_finance', 'role_crm')
ORDER BY r.name;
GO

PRINT '';
PRINT '════ VÉRIFICATION 3 : Table AppUsers ════';
SELECT
    login_name,
    role_name,
    full_name,
    email,
    is_active,
    created_at
FROM dbo.AppUsers
ORDER BY role_name;
GO

PRINT '';
PRINT '════ VÉRIFICATION 4 : Permissions par rôle et par table ════';
SELECT
    r.name                  AS role_sql,
    o.name                  AS table_ou_vue,
    p.permission_name       AS permission,
    p.state_desc            AS statut
FROM sys.database_role_members  drm
JOIN sys.database_principals    r  ON r.principal_id  = drm.role_principal_id
JOIN sys.database_permissions   p  ON p.grantee_principal_id = r.principal_id
JOIN sys.objects                o  ON o.object_id = p.major_id
WHERE r.name IN ('role_marketing', 'role_finance', 'role_crm')
  AND p.permission_name = 'SELECT'
ORDER BY r.name, o.name;
GO

PRINT '';
PRINT '════════════════════════════════════════════════════════════';
PRINT ' Script EventZilla terminé avec succès.';
PRINT ' Logins à utiliser dans Streamlit / FastAPI / Power BI :';
PRINT '   ranim_chikhrouhou  /  Ranim@Marketing2025!';
PRINT '   naima_sarraj       /  Naima@Finance2025!';
PRINT '   anas_allam         /  Anas@CRM2025!';
PRINT '════════════════════════════════════════════════════════════';
GO
