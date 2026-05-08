# EventZilla DWH — Per–fact visualization (Mermaid, detailed)

Each block below is a **full diagram**: one fact table and all its dimensions with **column details**.  
**Facts = pink** · **Dimensions = purple**.  
Copy a block into [https://mermaid.live](https://mermaid.live).

---

## Block 1 — Fact_CustomerSatisfaction (full detail)

**Raw measures:** `rating`, `final_price`.

```mermaid
graph TB
  DimDate["DimDate<br/>id_date PK<br/>full_date date<br/>day, month, year, quarter int<br/>day_of_week varchar<br/>is_weekend boolean<br/>is_holiday boolean<br/>holiday_name, holiday_local_name<br/>holiday_type, is_national"]

  DimReservation["DimReservation<br/>id_reservation PK<br/>reservation_date date<br/>status varchar"]

  DimProvider["DimProvider<br/>id_provider PK<br/>provider_name, service_type<br/>email, phone, city varchar"]

  DimFeedback["DimFeedback<br/>id_feedback PK<br/>comment varchar"]

  DimComplaint["DimComplaint<br/>id_complaint PK<br/>subject, description<br/>status varchar"]

  Fact["Fact_CustomerSatisfaction<br/>fact_satisfaction_id PK<br/>id_date FK<br/>id_reservation FK<br/>id_provider FK<br/>id_feedback FK<br/>id_complaint FK<br/>rating int<br/>final_price float"]

  DimDate -->|id_date| Fact
  DimReservation -->|id_reservation| Fact
  DimProvider -->|id_provider| Fact
  DimFeedback -->|id_feedback| Fact
  DimComplaint -->|id_complaint| Fact

  classDef fact fill:#FFB6C1,stroke:#C2185B,color:#333
  classDef dimension fill:#E1BEE7,stroke:#7B1FA2,color:#333
  class Fact fact
  class DimDate,DimReservation,DimProvider,DimFeedback,DimComplaint dimension
```

---

## Block 2 — Fact_CommercialPerformance (full detail)

**Raw measures:** `nb_visitors`, `nb_reservations_site`, `final_price`, `event_budget`, `service_price`.

```mermaid
graph TB
  DimDate["DimDate<br/>id_date PK<br/>full_date date<br/>day, month, year, quarter int<br/>day_of_week varchar<br/>is_weekend boolean<br/>is_holiday boolean<br/>holiday_name, holiday_local_name<br/>holiday_type, is_national"]

  DimEvent["DimEvent<br/>id_event PK<br/>title, event_type varchar<br/>event_date date"]

  DimReservation["DimReservation<br/>id_reservation PK<br/>reservation_date date<br/>status varchar"]

  DimBeneficiary["DimBeneficiary<br/>id_beneficiary PK<br/>first_name, last_name<br/>email, phone varchar"]

  DimServiceCategory["DimServiceCategory<br/>id_servicecategory PK<br/>service_title varchar<br/>subcategory_name, category_name varchar"]

  DimProvider["DimProvider<br/>id_provider PK<br/>provider_name, service_type<br/>email, phone, city varchar"]

  DimVisitors["DimVisitors<br/>id_visitors PK<br/>visit_date date"]

  Fact["Fact_CommercialPerformance<br/>fact_marketing_id PK<br/>id_date FK, id_event FK<br/>id_reservation FK, id_beneficiary FK<br/>id_servicecategory FK, id_provider FK<br/>id_visitors FK<br/>nb_visitors int<br/>nb_reservations_site int<br/>final_price float<br/>event_budget float<br/>service_price float"]

  DimDate -->|id_date| Fact
  DimEvent -->|id_event| Fact
  DimReservation -->|id_reservation| Fact
  DimBeneficiary -->|id_beneficiary| Fact
  DimServiceCategory -->|id_servicecategory| Fact
  DimProvider -->|id_provider| Fact
  DimVisitors -->|id_visitors| Fact

  classDef fact fill:#FFB6C1,stroke:#C2185B,color:#333
  classDef dimension fill:#E1BEE7,stroke:#7B1FA2,color:#333
  class Fact fact
  class DimDate,DimEvent,DimReservation,DimBeneficiary,DimServiceCategory,DimProvider,DimVisitors dimension
```

---

## Block 3 — Fact_FinancialProfitability (full detail)

**Raw measures:** `final_price`, `service_price`, `benchmark_avg_price`, `event_budget`.

```mermaid
graph TB
  DimDate["DimDate<br/>id_date PK<br/>full_date date<br/>day, month, year, quarter int<br/>day_of_week varchar<br/>is_weekend boolean<br/>is_holiday boolean<br/>holiday_name, holiday_local_name<br/>holiday_type, is_national"]

  DimEvent["DimEvent<br/>id_event PK<br/>title, event_type varchar<br/>event_date date"]

  DimReservation["DimReservation<br/>id_reservation PK<br/>reservation_date date<br/>status varchar"]

  DimServiceCategory["DimServiceCategory<br/>id_servicecategory PK<br/>service_title varchar<br/>subcategory_name, category_name varchar"]

  DimBenchmarkPrice["DimBenchmarkPrice<br/>id_benchmark PK<br/>subcategory varchar<br/>seasonality, period varchar"]

  DimProvider["DimProvider<br/>id_provider PK<br/>provider_name, service_type<br/>email, phone, city varchar"]

  Fact["Fact_FinancialProfitability<br/>fact_finance_id PK<br/>id_date FK, id_event FK<br/>id_reservation FK, id_servicecategory FK<br/>id_benchmark FK, id_provider FK<br/>final_price float<br/>service_price float<br/>benchmark_avg_price float<br/>event_budget float"]

  DimDate -->|id_date| Fact
  DimEvent -->|id_event| Fact
  DimReservation -->|id_reservation| Fact
  DimServiceCategory -->|id_servicecategory| Fact
  DimBenchmarkPrice -->|id_benchmark| Fact
  DimProvider -->|id_provider| Fact

  classDef fact fill:#FFB6C1,stroke:#C2185B,color:#333
  classDef dimension fill:#E1BEE7,stroke:#7B1FA2,color:#333
  class Fact fact
  class DimDate,DimEvent,DimReservation,DimServiceCategory,DimBenchmarkPrice,DimProvider dimension
```

---

## Relationship summary

| Fact | Keys to dimensions |
|------|--------------------|
| **Fact_CustomerSatisfaction** | id_date → DimDate · id_reservation → DimReservation · id_provider → DimProvider · id_feedback → DimFeedback · id_complaint → DimComplaint |
| **Fact_CommercialPerformance** | id_date → DimDate · id_event → DimEvent · id_reservation → DimReservation · id_beneficiary → DimBeneficiary · id_servicecategory → DimServiceCategory · id_provider → DimProvider · id_visitors → DimVisitors |
| **Fact_FinancialProfitability** | id_date → DimDate · id_event → DimEvent · id_reservation → DimReservation · id_servicecategory → DimServiceCategory · id_benchmark → DimBenchmarkPrice · id_provider → DimProvider |

**Colour legend:** Fact tables = **pink** (#FFB6C1) · Dimensions = **purple** (#E1BEE7).
