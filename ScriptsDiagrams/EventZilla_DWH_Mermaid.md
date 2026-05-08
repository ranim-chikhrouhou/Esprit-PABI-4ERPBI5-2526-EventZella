# EventZilla Data Warehouse — 3-Level Star Model

Modelling aligned with **Diagramme_BI_EventZella_3Niveaux** (PlantUML).  
**N1** = Time dimension · **N2** = Fact tables (raw measures) · **N3** = Analytical dimensions.  
KPIs are calculated in the BI layer, not in the DWH.

**Tools:**
- **dbdiagram.io**: [https://dbdiagram.io/d](https://dbdiagram.io/d) → Import → DBML → paste `EventZilla_DWH_Model.dbml`
- **Mermaid**: copy the block below into [https://mermaid.live](https://mermaid.live)

---

## Complete Mermaid schema with colours (detailed flowchart)

**Facts = pink** · **Dimensions = purple**. Full diagram with column details. Copy this block into [mermaid.live](https://mermaid.live).

```mermaid
graph TB
  subgraph N1["N1: Time dimension"]
    DimDate["DimDate<br/>id_date PK<br/>full_date date<br/>day, month, year, quarter int<br/>day_of_week varchar<br/>is_weekend boolean<br/>is_holiday boolean<br/>holiday_name, holiday_local_name<br/>holiday_type varchar<br/>is_national boolean"]
  end

  subgraph N2["N2: Fact tables"]
    Fact_Sat["Fact_CustomerSatisfaction<br/>fact_satisfaction_id PK<br/>id_date FK, id_reservation FK<br/>id_provider FK, id_feedback FK<br/>id_complaint FK<br/>rating int<br/>final_price float"]
    Fact_Perf["Fact_CommercialPerformance<br/>fact_marketing_id PK<br/>id_date FK, id_event FK<br/>id_reservation FK, id_beneficiary FK<br/>id_servicecategory FK, id_provider FK<br/>id_visitors FK<br/>nb_visitors, nb_reservations_site int<br/>final_price, event_budget<br/>service_price float"]
    Fact_Fin["Fact_FinancialProfitability<br/>fact_finance_id PK<br/>id_date FK, id_event FK<br/>id_reservation FK, id_servicecategory FK<br/>id_benchmark FK, id_provider FK<br/>final_price, service_price float<br/>benchmark_avg_price float<br/>event_budget float"]
  end

  subgraph N3["N3: Analytical dimensions"]
    DimFeedback["DimFeedback<br/>id_feedback PK<br/>comment varchar"]
    DimComplaint["DimComplaint<br/>id_complaint PK<br/>subject, description, status varchar"]
    DimProvider["DimProvider<br/>id_provider PK<br/>provider_name, service_type<br/>email, phone, city varchar"]
    DimVisitors["DimVisitors<br/>id_visitors PK<br/>visit_date date"]
    DimEvent["DimEvent<br/>id_event PK<br/>title, event_type varchar<br/>event_date date"]
    DimReservation["DimReservation<br/>id_reservation PK<br/>reservation_date date<br/>status varchar"]
    DimBeneficiary["DimBeneficiary<br/>id_beneficiary PK<br/>first_name, last_name<br/>email, phone varchar"]
    DimServiceCategory["DimServiceCategory<br/>id_servicecategory PK<br/>service_title varchar<br/>subcategory_name, category_name varchar"]
    DimBenchmarkPrice["DimBenchmarkPrice<br/>id_benchmark PK<br/>subcategory, seasonality<br/>period varchar"]
  end

  DimDate --> Fact_Sat
  DimDate --> Fact_Perf
  DimDate --> Fact_Fin
  Fact_Sat --> DimFeedback
  Fact_Sat --> DimComplaint
  Fact_Sat --> DimProvider
  Fact_Sat --> DimReservation
  Fact_Perf --> DimEvent
  Fact_Perf --> DimReservation
  Fact_Perf --> DimBeneficiary
  Fact_Perf --> DimServiceCategory
  Fact_Perf --> DimProvider
  Fact_Perf --> DimVisitors
  Fact_Fin --> DimEvent
  Fact_Fin --> DimReservation
  Fact_Fin --> DimServiceCategory
  Fact_Fin --> DimBenchmarkPrice
  Fact_Fin --> DimProvider

  classDef fact fill:#FFB6C1,stroke:#C2185B,color:#333
  classDef dimension fill:#E1BEE7,stroke:#7B1FA2,color:#333
  class Fact_Sat,Fact_Perf,Fact_Fin fact
  class DimDate,DimFeedback,DimComplaint,DimProvider,DimVisitors,DimEvent,DimReservation,DimBeneficiary,DimServiceCategory,DimBenchmarkPrice dimension
```

---

## Mermaid ER schema — same detail, ER syntax

Same columns as above in ER format. ER diagrams do not support fact/dimension colours; use the detailed flowchart above for colours.

```mermaid
erDiagram
  DimDate ||--o{ Fact_CustomerSatisfaction : id_date
  DimDate ||--o{ Fact_CommercialPerformance : id_date
  DimDate ||--o{ Fact_FinancialProfitability : id_date

  Fact_CustomerSatisfaction }o--|| DimFeedback : id_feedback
  Fact_CustomerSatisfaction }o--|| DimComplaint : id_complaint
  Fact_CustomerSatisfaction }o--|| DimProvider : id_provider
  Fact_CustomerSatisfaction }o--|| DimReservation : id_reservation

  Fact_CommercialPerformance }o--|| DimEvent : id_event
  Fact_CommercialPerformance }o--|| DimReservation : id_reservation
  Fact_CommercialPerformance }o--|| DimBeneficiary : id_beneficiary
  Fact_CommercialPerformance }o--|| DimServiceCategory : id_servicecategory
  Fact_CommercialPerformance }o--|| DimProvider : id_provider
  Fact_CommercialPerformance }o--|| DimVisitors : id_visitors

  Fact_FinancialProfitability }o--|| DimEvent : id_event
  Fact_FinancialProfitability }o--|| DimReservation : id_reservation
  Fact_FinancialProfitability }o--|| DimServiceCategory : id_servicecategory
  Fact_FinancialProfitability }o--|| DimBenchmarkPrice : id_benchmark
  Fact_FinancialProfitability }o--|| DimProvider : id_provider

  DimDate {
    int id_date PK
    date full_date
    int day
    int month
    int year
    int quarter
    varchar day_of_week
    boolean is_weekend
    boolean is_holiday
    varchar holiday_name
    varchar holiday_local_name
    varchar holiday_type
    boolean is_national
  }

  Fact_CustomerSatisfaction {
    int fact_satisfaction_id PK
    int id_date FK
    int id_reservation FK
    int id_provider FK
    int id_feedback FK
    int id_complaint FK
    int rating
    float final_price
  }

  Fact_CommercialPerformance {
    int fact_marketing_id PK
    int id_date FK
    int id_event FK
    int id_reservation FK
    int id_beneficiary FK
    int id_servicecategory FK
    int id_provider FK
    int id_visitors FK
    int nb_visitors
    int nb_reservations_site
    float final_price
    float event_budget
    float service_price
  }

  Fact_FinancialProfitability {
    int fact_finance_id PK
    int id_date FK
    int id_event FK
    int id_reservation FK
    int id_servicecategory FK
    int id_benchmark FK
    int id_provider FK
    float final_price
    float service_price
    float benchmark_avg_price
    float event_budget
  }

  DimFeedback {
    int id_feedback PK
    varchar comment
  }

  DimComplaint {
    int id_complaint PK
    varchar subject
    varchar description
    varchar status
  }

  DimProvider {
    int id_provider PK
    varchar provider_name
    varchar service_type
    varchar email
    varchar phone
    varchar city
  }

  DimVisitors {
    int id_visitors PK
    date visit_date
  }

  DimEvent {
    int id_event PK
    varchar title
    varchar event_type
    date event_date
  }

  DimReservation {
    int id_reservation PK
    date reservation_date
    varchar status
  }

  DimBeneficiary {
    int id_beneficiary PK
    varchar first_name
    varchar last_name
    varchar email
    varchar phone
  }

  DimServiceCategory {
    int id_servicecategory PK
    varchar service_title
    varchar subcategory_name
    varchar category_name
  }

  DimBenchmarkPrice {
    int id_benchmark PK
    varchar subcategory
    varchar seasonality
    varchar period
  }
```

---

## Legend (aligned with PlantUML)

| Level | Type | Tables |
|--------|------|--------|
| **N1** | Time dimension | DimDate (merged with external_holidays.csv) |
| **N2** | Fact tables | Fact_CustomerSatisfaction, Fact_CommercialPerformance, Fact_FinancialProfitability |
| **N3** | Analytical dimensions | DimFeedback, DimComplaint, DimProvider, DimVisitors, DimEvent, DimReservation, DimBeneficiary, DimServiceCategory, DimBenchmarkPrice |

**Raw measures (in fact tables only):**

| Fact | Measures | ETL source |
|------|----------|------------|
| Fact_CustomerSatisfaction | rating, final_price | DimFeedback, DimReservation |
| Fact_CommercialPerformance | nb_visitors, nb_reservations_site, final_price, event_budget, service_price | DimVisitors, DimReservation, DimEvent, DimServiceCategory |
| Fact_FinancialProfitability | final_price, service_price, benchmark_avg_price, event_budget | DimReservation, DimServiceCategory, DimBenchmarkPrice, DimEvent |

**KPIs (calculated in BI, not in DWH):** Conversion rate, Average basket, Total revenue, Average rating, Resolution rate, Share below/aligned/above market, Holiday impact on revenue, Commission rate, etc.

**Data sources:** DimDate ← Dates + external_holidays.csv · DimFeedback ← EVALUATION · DimVisitors ← VISITORS · DimEvent ← EVENT · DimReservation ← RESERVATION · DimServiceCategory ← SERVICE/SUBCAT/CAT · DimBenchmarkPrice ← benchmark CSV
