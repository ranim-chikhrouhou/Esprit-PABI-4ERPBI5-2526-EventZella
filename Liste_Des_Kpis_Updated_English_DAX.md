# EventZilla — KPI catalogue & DAX measures (English)

**Source document:** `Liste Des KpisUpdatedEnglish.pdf` (export structured for version control).

**Semantic model:** align table and column names with your Power BI / warehouse model (`Fact_PerformanceCommerciale`, `Fact_RentabiliteFinanciere`, `Fact_SatisfactionClient`, `DimReservation`, `DimDate`, `DimComplaint`, `DimMarketingSpend`, `DimTendanceMarche`, `DimVenue`, `DimServiceCategory`, `DimBeneficiary`, etc.).

**Note:** Some measures reference other measures (e.g. `[Number of Reservations]`). Create dependency measures first.

---

## Table of contents

1. [Venues & catalog](#1-venues--catalog)
2. [Reservations & conversion](#2-reservations--conversion)
3. [Revenue & commissions](#3-revenue--commissions)
4. [Market positioning (benchmark)](#4-market-positioning-benchmark)
5. [Satisfaction & complaints](#5-satisfaction--complaints)
6. [Marketing & LTV](#6-marketing--ltv)
7. [Holidays & time-based revenue](#7-holidays--time-based-revenue)
8. [Advanced retention (month-over-month)](#8-advanced-retention-month-over-month)

---

## 1. Venues & catalog

### % Available Contactable Venues

```dax
% Available Contactable Venues =
DIVIDE (
    COUNTROWS ( FILTER ( 'DimVenue', NOT ISBLANK ( 'DimVenue'[contact] ) ) ),
    COUNTROWS ( 'DimVenue' ),
    0
) * 100
```

### Number of Available venues with contact

```dax
Number of Available venues with contact = COUNTROWS ( DimVenue )
```

### Number of Event categories

```dax
Number of Event categories =
DISTINCTCOUNT ( 'DimServiceCategory'[category_name] )
```

### Number Of potential Event Categories

```dax
Number Of potential Event Categories =
VAR CategoriesMarche = VALUES ( DimTendanceMarche[category_name] )
VAR CategoriesEventZilla = VALUES ( DimServiceCategory[category_name] )
RETURN
    COUNTROWS ( EXCEPT ( CategoriesMarche, CategoriesEventZilla ) )
```

### Event Catalog Diversity %

```dax
Event Catalog Diversity % =
VAR Dist =
    CALCULATE (
        DISTINCTCOUNT ( DimServiceCategory[category_name] ),
        Fact_PerformanceCommerciale
    )
VAR Tot =
    COUNTROWS ( ALL ( DimServiceCategory[category_name] ) )
RETURN
    DIVIDE ( Dist, Tot, 0 ) * 100
```

### Rank Of Market Opportunities

```dax
Rank Of Market Opportunities =
IF (
    [Number Of potential Event Categories] > 0,
    RANKX (
        ALL ( DimTendanceMarche[category_name] ),
        CALCULATE ( MAX ( DimTendanceMarche[event_count_observed] ) ),,
        DESC,
        DENSE
    ),
    BLANK ()
)
```

### Top N Event Categories To add

```dax
Top N Event Categories To add =
VAR TopSelectionne = 5   // Adjust as needed
RETURN
    IF (
        [Rank Of Market Opportunities] <= TopSelectionne,
        [Number Of potential Event Categories],
        BLANK ()
    )
```

---

## 2. Reservations & conversion

### Number of Reservations

```dax
Number of Reservations = COUNT ( DimReservation[id_reservation] )
```

### Acceptance Rate %

```dax
Acceptance Rate % =
DIVIDE (
    CALCULATE ( COUNT ( DimReservation[id_reservation] ), DimReservation[status] = "confirmed" ),
    COUNT ( DimReservation[id_reservation] ),
    0
)
```

### Cancellation Rate %

```dax
Cancellation Rate % =
DIVIDE (
    CALCULATE ( COUNT ( DimReservation[id_reservation] ), DimReservation[status] = "cancelled" ),
    COUNT ( DimReservation[id_reservation] ),
    0
)
```

### Conversion Rate

```dax
Conversion Rate =
DIVIDE (
    SUM ( Fact_PerformanceCommerciale[nb_reservations_site] ),
    SUM ( Fact_PerformanceCommerciale[nb_visitors] ),
    0
)
```

### Number of visitors

```dax
Number of visitors = SUM ( Fact_PerformanceCommerciale[nb_visitors] )
```

### Reservation Frequency / Beneficiary

```dax
Reservation Frequency / Beneficiary =
DIVIDE (
    CALCULATE (
        DISTINCTCOUNT ( Fact_PerformanceCommerciale[id_reservation] ),
        DimReservation[status] = "confirmed"
    ),
    DISTINCTCOUNT ( Fact_PerformanceCommerciale[id_beneficiary] ),
    BLANK ()
)
```

### Reservations Rate During Holidays

```dax
Reservations Rate During Holidays =
DIVIDE (
    CALCULATE ( [Number of Reservations], DimDate[is_holiday] = TRUE () ),
    [Number of Reservations],
    0
)
```

---

## 3. Revenue & commissions

### Total Confirmed Revenue

```dax
Total Confirmed Revenue =
CALCULATE (
    SUM ( Fact_RentabiliteFinanciere[final_price] ),
    DimReservation[status] = "confirmed"
)
```

### Total Revenue (TND)

```dax
Total Revenue (TND) =
CALCULATE (
    SUM ( Fact_RentabiliteFinanciere[final_price] ),
    DimReservation[status] = "confirmed"
)
```

### Average Order Value (TND)

```dax
Average Order Value (TND) =
CALCULATE (
    AVERAGE ( Fact_RentabiliteFinanciere[final_price] ),
    DimReservation[status] = "confirmed"
)
```

### Average Reservation Price EventZilla

```dax
Average Reservation Price EventZilla =
AVERAGE ( Fact_RentabiliteFinanciere[final_price] )
```

### Commissions (TND)

```dax
Commissions (TND) =
CALCULATE (
    SUMX (
        'Fact_RentabiliteFinanciere',
        'Fact_RentabiliteFinanciere'[final_price] - 'Fact_RentabiliteFinanciere'[service_price]
    ),
    'DimReservation'[status] = "confirmed"
)
```

### Commission Rate %

```dax
Commission Rate % =
VAR MargeTotale = [Commissions (TND)]
VAR CA_Confirme =
    CALCULATE (
        SUM ( 'Fact_RentabiliteFinanciere'[final_price] ),
        'DimReservation'[status] = "confirmed"
    )
RETURN
    DIVIDE ( MargeTotale, CA_Confirme, 0 ) * 100
```

---

## 4. Market positioning (benchmark)

### Part Below Market

```dax
Part Below Market =
VAR TotalConfirme =
    CALCULATE ( COUNTROWS ( Fact_RentabiliteFinanciere ), DimReservation[status] = "confirmed" )
VAR SousMarche =
    CALCULATE (
        COUNTROWS ( Fact_RentabiliteFinanciere ),
        DimReservation[status] = "confirmed",
        Fact_RentabiliteFinanciere[final_price]
            < 0.85 * Fact_RentabiliteFinanciere[benchmark_avg_price]
    )
RETURN
    DIVIDE ( SousMarche, TotalConfirme, 0 )
```

### Part Aligned with Market

```dax
Part Aligned with Market =
VAR TotalConfirme =
    CALCULATE ( COUNTROWS ( Fact_RentabiliteFinanciere ), DimReservation[status] = "confirmed" )
VAR Aligne =
    CALCULATE (
        COUNTROWS ( Fact_RentabiliteFinanciere ),
        DimReservation[status] = "confirmed",
        Fact_RentabiliteFinanciere[final_price] >= 0.85 * Fact_RentabiliteFinanciere[benchmark_avg_price]
            && Fact_RentabiliteFinanciere[final_price] <= 1.15 * Fact_RentabiliteFinanciere[benchmark_avg_price]
    )
RETURN
    DIVIDE ( Aligne, TotalConfirme, 0 )
```

### Part Above Market

```dax
Part Above Market =
VAR TotalConfirme =
    CALCULATE ( COUNTROWS ( Fact_RentabiliteFinanciere ), DimReservation[status] = "confirmed" )
VAR AuDessus =
    CALCULATE (
        COUNTROWS ( Fact_RentabiliteFinanciere ),
        DimReservation[status] = "confirmed",
        Fact_RentabiliteFinanciere[final_price] > 1.15 * Fact_RentabiliteFinanciere[benchmark_avg_price]
    )
RETURN
    DIVIDE ( AuDessus, TotalConfirme, 0 )
```

---

## 5. Satisfaction & complaints

### Average Providers Rate

```dax
Average Providers Rate = AVERAGE ( 'Fact_SatisfactionClient'[rating] )
```

### NPS (Net Promoter Score)

```dax
NPS (Net Promoter Score) =
VAR Total = COUNTROWS ( Fact_SatisfactionClient )
VAR Promoteurs =
    CALCULATE ( COUNTROWS ( Fact_SatisfactionClient ), Fact_SatisfactionClient[rating] >= 4 )
VAR Detracteurs =
    CALCULATE ( COUNTROWS ( Fact_SatisfactionClient ), Fact_SatisfactionClient[rating] <= 2 )
RETURN
    ( DIVIDE ( Promoteurs, Total, 0 ) - DIVIDE ( Detracteurs, Total, 0 ) ) * 100
```

### Total Number Of Complaints

```dax
Total Number Of Complaints = COUNTROWS ( 'DimComplaint' )
```

### Complaints Rate / 100 reservations

```dax
Complaints Rate / 100 reservations =
DIVIDE (
    COUNTROWS ( 'DimComplaint' ),
    DISTINCTCOUNT ( 'DimReservation'[id_reservation] ),
    0
) * 100
```

### Complaints resolution rate

> **Implementation note:** The PDF text references `RELATEDTABLE ( 'Fact_SatisfactionClient' )`, which does not match a typical complaints fact on `DimComplaint`. The version below follows the same logic as the French KPI list (closed complaints / all complaints). **Validate against your model** and adjust table/column names if your semantic model differs.

```dax
Complaints resolution rate =
DIVIDE (
    CALCULATE ( COUNTROWS ( DimComplaint ), DimComplaint[status] = "closed" ),
    COUNTROWS ( DimComplaint ),
    0
)
```

---

## 6. Marketing & LTV

### Total_Marketing

```dax
Total_Marketing = SUM ( DimMarketingSpend[marketing_spend] )
```

### Customer Acquisition Cost (TND / Beneficiary)

```dax
Customer Acquisition Cost (TND / Beneficiary) =
DIVIDE (
    SUM ( DimMarketingSpend[marketing_spend] ),
    SUM ( DimMarketingSpend[new_beneficiaries] ),
    0
)
```

### ROI (Return on Investment) Marketing

```dax
ROI (Return on Investment) Marketing =
DIVIDE ( [Total Confirmed Revenue], [Total_Marketing] )
```

### Recurring Beneficiaries

```dax
Recurring Beneficiaries =
COUNTROWS (
    FILTER (
        VALUES ( DimBeneficiary[id_beneficiary] ),
        CALCULATE ( COUNT ( DimReservation[id_reservation] ), DimReservation[status] = "confirmed" ) > 1
    )
)
```

### Beneficiairies Retention Rate % *(spelling per source PDF)*

```dax
Beneficiairies Retention Rate % =
DIVIDE (
    [Recurring Beneficiaries],
    DISTINCTCOUNT ( DimBeneficiary[id_beneficiary] ),
    BLANK ()
) * 100
```

### LTV (Life Time value) simplified

```dax
LTV (Life Time value) simplified =
    [Average Order Value (TND)] * [Reservation Frequency / Beneficiary]
```

---

## 7. Holidays & time-based revenue

### Revenue With Holidays

```dax
Revenue With Holidays =
CALCULATE (
    SUM ( 'Fact_RentabiliteFinanciere'[final_price] ),
    'DimDate'[is_holiday] = TRUE ()
)
```

### Revenue Without Holidays

```dax
Revenue Without Holidays =
CALCULATE (
    SUM ( 'Fact_RentabiliteFinanciere'[final_price] ),
    'DimDate'[is_holiday] = FALSE ()
)
```

### Holidays Affect Difference

```dax
Holidays Affect Difference = [Revenue With Holidays] - [Revenue Without Holidays]
```

### Revenue holidays/NoHolidays Difference

*(Same definition as Holidays Affect Difference in the source PDF.)*

```dax
Revenue holidays/NoHolidays Difference = [Revenue With Holidays] - [Revenue Without Holidays]
```

---

## 8. Advanced retention (month-over-month)

### Retention Rate %

```dax
Retention Rate % =
VAR MoisActuel = SELECTEDVALUE ( 'DimDate'[month] )
VAR AnneeActuelle = SELECTEDVALUE ( 'DimDate'[year] )
VAR MoisPrecedent = IF ( MoisActuel = 1, 12, MoisActuel - 1 )
VAR AnneePrecedente = IF ( MoisActuel = 1, AnneeActuelle - 1, AnneeActuelle )
VAR ListeClientsCeMois =
    CALCULATETABLE (
        VALUES ( DimBeneficiary[id_beneficiary] ),
        Fact_SatisfactionClient,
        'DimDate'[month] = MoisActuel,
        'DimDate'[year] = AnneeActuelle,
        ALL ( 'DimDate' )
    )
VAR ListeClientsMoisDernier =
    CALCULATETABLE (
        VALUES ( DimBeneficiary[id_beneficiary] ),
        Fact_SatisfactionClient,
        'DimDate'[month] = MoisPrecedent,
        'DimDate'[year] = AnneePrecedente,
        ALL ( 'DimDate' )
    )
VAR ClientsRetenus = COUNTROWS ( INTERSECT ( ListeClientsCeMois, ListeClientsMoisDernier ) )
VAR TotalCeMois = COUNTROWS ( ListeClientsCeMois )
RETURN
    DIVIDE ( ClientsRetenus, TotalCeMois, 0 )
```

---

## Measure inventory (quick reference)

| # | Measure name |
|---|----------------|
| 1 | % Available Contactable Venues |
| 2 | Acceptance Rate % |
| 3 | Average Order Value (TND) |
| 4 | Average Providers Rate |
| 5 | Average Reservation Price EventZilla |
| 6 | Beneficiairies Retention Rate % |
| 7 | Cancellation Rate % |
| 8 | Commission Rate % |
| 9 | Commissions (TND) |
| 10 | Complaints Rate / 100 reservations |
| 11 | Complaints resolution rate |
| 12 | Conversion Rate |
| 13 | Customer Acquisition Cost (TND / Beneficiary) |
| 14 | Event Catalog Diversity % |
| 15 | Holidays Affect Difference |
| 16 | LTV (Life Time value) simplified |
| 17 | NPS (Net Promoter Score) |
| 18 | Number of Available venues with contact |
| 19 | Number of Event categories |
| 20 | Number Of potential Event Categories |
| 21 | Number of Reservations |
| 22 | Number of visitors |
| 23 | Part Above Market |
| 24 | Part Aligned with Market |
| 25 | Part Below Market |
| 26 | Rank Of Market Opportunities |
| 27 | Recurring Beneficiaries |
| 28 | Reservation Frequency / Beneficiary |
| 29 | Reservations Rate During Holidays |
| 30 | Retention Rate % |
| 31 | Revenue holidays/NoHolidays Difference |
| 32 | Revenue With Holidays |
| 33 | Revenue Without Holidays |
| 34 | ROI (Return on Investment) Marketing |
| 35 | Top N Event Categories To add |
| 36 | Total Confirmed Revenue |
| 37 | Total Number Of Complaints |
| 38 | Total Revenue (TND) |
| 39 | Total_Marketing |

---

*End of document — suitable for Git versioning alongside `.pbix` exports and `README.md`.*
