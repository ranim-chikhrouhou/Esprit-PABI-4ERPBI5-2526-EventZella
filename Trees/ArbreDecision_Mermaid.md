# Decision Trees — 5 DESCRIPTIVE, 2 EXPLANATORY, 1 PREDICTIVE

Structure: **Global objective** → **5 DESCRIPTIVE** / **2 EXPLANATORY** / **1 PREDICTIVE** → Operational objectives → KPI.  
Copy a block into https://mermaid.live (from `graph TB` to the last `class` line).  
Colours: Global = sky blue, Descriptive = pink, Explanatory = purple, Predictive = green.

---

## BLOCK 1 — Marketing Manager

graph TB
  G[Global Objective - Optimize commercial performance]
  G --> D1
  G --> D2
  G --> D3
  G --> D4
  G --> D5
  G --> E1
  G --> E2
  G --> PRED
  D1[DESCRIPTIVE 1 - Describe current commercial performance]
  D1 --> D1a[Visualize total number of bookings]
  D1a --> D1a1[KPI Total number of bookings]
  D1 --> D1b[Visualize conversion rate]
  D1b --> D1b1[KPI Conversion rate]
  D1 --> D1c[Visualize number of visitors]
  D1c --> D1c1[KPI Number of visitors]
  D2[DESCRIPTIVE 2 - Describe diversity of booked categories]
  D2 --> D2a[Visualize diversity of booked categories]
  D2a --> D2a1[KPI Diversity of booked categories]
  D3[DESCRIPTIVE 3 - Describe top N categories to add]
  D3 --> D3a[Identify top N categories to add]
  D3a --> D3a1[KPI Top N categories to add]
  D4[DESCRIPTIVE 4 - Describe geographic distribution]
  D4 --> D4a[Visualize geographic distribution]
  D4a --> D4a1[KPI Venue distribution by governorate]
  D5[DESCRIPTIVE 5 - Describe service category distribution]
  D5 --> D5a[Visualize diversity top N and distribution]
  D5a --> D5a1[KPI Diversity Top N Governorate distribution]
  E1[EXPLANATORY 1 - Explain conversion, campaigns, LTV and retention]
  E1 --> E1a[Analyze conversion vs acceptance, cancellation and visitors]
  E1a --> E1a1[KPI Conversion acceptance cancellation visitors]
  E1 --> E1b[Analyze campaigns: CAC, bookings, conversion and visitors]
  E1b --> E1b1[KPI CAC Bookings Conversion Visitors]
  E1 --> E1c[Understand relationship between LTV CAC retention and customer value]
  E1c --> E1c1[KPI LTV CAC Retention Bookings]
  E1 --> E1d[Analyze retention vs cancellation and category factors]
  E1d --> E1d1[KPI Retention Cancellation New vs covered categories]
  E2[EXPLANATORY 2 - Explain pricing, growth, holidays and geographic coverage]
  E2 --> E2a[Analyze pricing below aligned above market by bookings]
  E2a --> E2a1[KPI Share below market aligned above Bookings Conversion]
  E2 --> E2b[Analyze growth via top N new categories and diversity]
  E2b --> E2b1[KPI Top N New covered categories Diversity]
  E2 --> E2c[Analyze holiday booking rate vs bookings and conversion]
  E2c --> E2c1[KPI Holiday booking rate Bookings Conversion]
  E2 --> E2d[Analyze coverage: under-served areas, distribution and conversion]
  E2d --> E2d1[KPI Governorate distribution Suggestable venues Bookings Conversion]
  PRED[PREDICTIVE - Predict future commercial performance]
  PRED --> P1[Predict future number of bookings]
  P1 --> P1k[KPI Total bookings projection]
  PRED --> P2[Predict future conversion rate]
  P2 --> P2k[KPI Conversion rate projection]
  PRED --> P3[Predict retention rate evolution]
  P3 --> P3k[KPI Beneficiary retention rate projection]
  PRED --> P4[Predict CAC and LTV evolution]
  P4 --> P4k[KPI CAC LTV projection]
  PRED --> P5[Predict holiday booking rate]
  P5 --> P5k[KPI Holiday booking rate projection]
  classDef global fill:#87CEEB,stroke:#4682B4
  classDef descriptive fill:#FFB6C1,stroke:#C2185B
  classDef explanatory fill:#E1BEE7,stroke:#7B1FA2
  classDef predictive fill:#C8E6C9,stroke:#2E7D32
  class G global
  class D1,D2,D3,D4,D5,D1a,D1b,D1c,D1a1,D1b1,D1c1,D2a,D2a1,D3a,D3a1,D4a,D4a1,D5a,D5a1 descriptive
  class E1,E2,E1a,E1b,E1c,E1d,E1a1,E1b1,E1c1,E1d1,E2a,E2b,E2c,E2d,E2a1,E2b1,E2c1,E2d1 explanatory
  class PRED,P1,P2,P3,P4,P5,P1k,P2k,P3k,P4k,P5k predictive

---

## BLOCK 2 — Finance Manager

graph TB
  G[Global Objective - Optimize profitability]
  G --> D1
  G --> D2
  G --> D3
  G --> D4
  G --> D5
  G --> E1
  G --> E2
  G --> PRED
  D1[DESCRIPTIVE 1 - Describe current revenue and financial performance]
  D1 --> D1a[Visualize total revenue]
  D1a --> D1a1[KPI Total revenue]
  D1 --> D1b[Visualize average basket]
  D1b --> D1b1[KPI Average basket]
  D2[DESCRIPTIVE 2 - Describe top N categories to add]
  D2 --> D2a[Identify top N categories to add]
  D2a --> D2a1[KPI Top N categories to add]
  D3[DESCRIPTIVE 3 - Describe revenue and commission structure]
  D3 --> D3a[Visualize commission rate on booking]
  D3a --> D3a1[KPI Commission rate on booking]
  D3 --> D3b[Compare revenue by period]
  D3b --> D3b1[KPI Total revenue Average basket]
  D4[DESCRIPTIVE 4 - Describe holiday impact on revenue]
  D4 --> D4a[Visualize holiday impact on revenue]
  D4a --> D4a1[KPI Holiday impact on revenue]
  D5[DESCRIPTIVE 5 - Describe revenue and commission indicators]
  D5 --> D5a[Analyze revenue by period and commission]
  D5a --> D5a1[KPI Revenue Basket Commission]
  E1[EXPLANATORY 1 - Explain profitability, CAC and customer value]
  E1 --> E1a[Analyze profitability through LTV, CAC, basket and categories]
  E1a --> E1a1[KPI LTV CAC Basket Revenue New vs covered categories]
  E1 --> E1b[Analyze CAC vs revenue, conversion and basket]
  E1b --> E1b1[KPI CAC Revenue Conversion Basket]
  E1 --> E1c[Analyze LTV impact on revenue, basket and categories]
  E1c --> E1c1[KPI LTV Revenue Basket New vs covered categories]
  E2[EXPLANATORY 2 - Explain pricing, holidays, commissions and growth]
  E2 --> E2a[Analyze pricing below aligned above market vs bookings and revenue]
  E2a --> E2a1[KPI Share below market aligned above Revenue Basket]
  E2 --> E2b[Analyze holiday impact on revenue and basket]
  E2b --> E2b1[KPI Holiday impact Revenue Basket Income]
  E2 --> E2c[Analyze commission rate vs revenue, basket and categories]
  E2c --> E2c1[KPI Commission Revenue Basket New vs covered categories]
  E2 --> E2d[Analyze growth via top N new categories and profitability]
  E2d --> E2d1[KPI Top N Covered categories LTV CAC]
  PRED[PREDICTIVE - Predict future revenue and profitability]
  PRED --> P1[Predict revenue evolution]
  P1 --> P1k[KPI Total revenue projection]
  PRED --> P2[Predict average basket evolution]
  P2 --> P2k[KPI Average basket projection]
  PRED --> P3[Predict LTV and CAC evolution]
  P3 --> P3k[KPI LTV CAC projection]
  PRED --> P4[Predict future holiday impact on revenue]
  P4 --> P4k[KPI Holiday impact on revenue projection]
  classDef global fill:#87CEEB,stroke:#4682B4
  classDef descriptive fill:#FFB6C1,stroke:#C2185B
  classDef explanatory fill:#E1BEE7,stroke:#7B1FA2
  classDef predictive fill:#C8E6C9,stroke:#2E7D32
  class G global
  class D1,D2,D3,D4,D5,D1a,D1b,D1a1,D1b1,D2a,D2a1,D3a,D3b,D3a1,D3b1,D4a,D4a1,D5a,D5a1 descriptive
  class E1,E2,E1a,E1b,E1c,E1a1,E1b1,E1c1,E2a,E2b,E2c,E2d,E2a1,E2b1,E2c1,E2d1 explanatory
  class PRED,P1,P2,P3,P4,P1k,P2k,P3k,P4k predictive

---

## BLOCK 3 — Customer Relationship Manager

graph TB
  G[Global Objective - Improve customer relationship]
  G --> D1
  G --> D2
  G --> D3
  G --> D4
  G --> D5
  G --> E1
  G --> E2
  G --> PRED
  D1[DESCRIPTIVE 1 - Describe current satisfaction and customer relationship]
  D1 --> D1a[Visualize number of complaints]
  D1a --> D1a1[KPI Number of complaints]
  D1 --> D1b[Visualize cancellation rate]
  D1b --> D1b1[KPI Cancellation rate]
  D2[DESCRIPTIVE 2 - Describe average rating and complaint rate]
  D2 --> D2a[Compare average provider rating]
  D2a --> D2a1[KPI Average provider rating]
  D2 --> D2b[Visualize complaint rate per 100 bookings]
  D2b --> D2b1[KPI Complaint rate per 100 bookings]
  D3[DESCRIPTIVE 3 - Describe service quality and responsiveness]
  D3 --> D3a[Visualize complaint resolution rate]
  D3a --> D3a1[KPI Complaint resolution rate]
  D3 --> D3b[Compare NPS]
  D3b --> D3b1[KPI NPS]
  D4[DESCRIPTIVE 4 - Describe share of reachable venues]
  D4 --> D4a[Visualize share of reachable venues]
  D4a --> D4a1[KPI Share of reachable venues]
  D5[DESCRIPTIVE 5 - Describe satisfaction and service indicators]
  D5 --> D5a[Analyze complaints, resolution, NPS and accessibility]
  D5a --> D5a1[KPI Complaints Resolution NPS Reachable venues]
  E1[EXPLANATORY 1 - Explain satisfaction, complaints, retention and service quality]
  E1 --> E1a[Analyze satisfaction via provider rating, cancellation and complaints per 100 bookings]
  E1a --> E1a1[KPI Provider rating Complaints Cancellation Complaints 100 Rating]
  E1 --> E1b[Analyze complaints vs cancellation, rating and resolution]
  E1b --> E1b1[KPI Complaints Cancellation Rating Resolution]
  E1 --> E1c[Analyze retention vs complaints per 100 bookings and cancellation]
  E1c --> E1c1[KPI Complaints 100 Retention Cancellation Complaints]
  E1 --> E1d[Analyze service quality via complaint resolution and reachable venues]
  E1d --> E1d1[KPI Resolution Complaints Complaints 100 Reachable venues]
  E2[EXPLANATORY 2 - Explain pricing, NPS and accessibility]
  E2 --> E2a[Analyze pricing below aligned above market vs complaints and cancellation]
  E2a --> E2a1[KPI Share below market aligned above Complaints Cancellation]
  E2 --> E2b[Analyze NPS vs complaints, retention and rating]
  E2b --> E2b1[KPI Complaints Retention Rating NPS]
  E2 --> E2c[Analyze accessibility via reachable venues and complaints indicators]
  E2c --> E2c1[KPI Reachable venues Complaints 100 Complaints Cancellation]
  PRED[PREDICTIVE - Predict future satisfaction and customer relationship]
  PRED --> P1[Predict complaint rate per 100 bookings evolution]
  P1 --> P1k[KPI Complaint rate per 100 bookings projection]
  PRED --> P2[Predict number of complaints evolution]
  P2 --> P2k[KPI Number of complaints projection]
  PRED --> P3[Predict retention rate evolution]
  P3 --> P3k[KPI Beneficiary retention rate projection]
  PRED --> P4[Predict NPS evolution]
  P4 --> P4k[KPI NPS projection]
  classDef global fill:#87CEEB,stroke:#4682B4
  classDef descriptive fill:#FFB6C1,stroke:#C2185B
  classDef explanatory fill:#E1BEE7,stroke:#7B1FA2
  classDef predictive fill:#C8E6C9,stroke:#2E7D32
  class G global
  class D1,D2,D3,D4,D5,D1a,D1b,D1a1,D1b1,D2a,D2b,D2a1,D2b1,D3a,D3b,D3a1,D3b1,D4a,D4a1,D5a,D5a1 descriptive
  class E1,E2,E1a,E1b,E1c,E1d,E1a1,E1b1,E1c1,E1d1,E2a,E2b,E2c,E2a1,E2b1,E2c1 explanatory
  class PRED,P1,P2,P3,P4,P1k,P2k,P3k,P4k predictive
