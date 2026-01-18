# Case Study 3 — VoC Analytics Dashboard (Advanced / High-Level Spec)

A high-level, portfolio-oriented dashboard specification designed to demonstrate **product thinking**, **cross-functional communication**, and **analytics system design**. The dashboard operationalizes review mining outputs into a decision-making product for Product, Ops, Cabin, Engineering, and Customer Care.

---

## 0) One-sentence purpose

A decision dashboard that answers: **“What is driving customer dissatisfaction right now, where is it happening, how severe is it, and what initiatives are working?”**

---

## 1) Users, decisions, and operating cadence

### Primary users
- **Executive / Head of Customer Experience:** prioritization, investment decisions, governance
- **Product & Digital:** reliability and UX improvements (Wi‑Fi, app/booking, IFE platforms)
- **Cabin Ops:** service consistency, training/coaching
- **Airport / Ground Ops:** boarding, baggage, station-level process quality
- **Customer Care:** policy and service recovery, contact-driver alignment

### Decisions supported (examples)
- Select the **top 3–5 themes** to staff this quarter
- Identify **hotspots** (route, fleet type, station) for targeted fixes
- Track whether initiatives are improving both **VoC metrics** and **operational KPIs**
- Detect and respond to **issue spikes** within days/weeks, not months

### Cadence
- **Weekly:** monitoring + alert triage + initiative pulse check  
- **Monthly:** steering review + backlog reprioritization + governance updates  
- **Quarterly:** strategy and investment planning; taxonomy/model review

---

## 2) KPI framework (what “good” looks like)

### A) VoC health KPIs (top-level)
- **Overall negative rate**: share of reviews predicted negative
- **Top topic negative mention rate**: per topic, share of all reviews that are (topic mentioned AND negative)
- **Topic severity**: per topic, mean `p_neg` among mentions
- **Trend delta**: month-over-month and quarter-over-quarter changes

### B) Initiative KPIs (closed-loop)
Per initiative:
- Primary VoC KPI: topic negative mention rate (rolling 4-week or rolling 3-month)
- Secondary VoC KPI: topic severity (avg `p_neg`)
- Operational KPI(s) linked to initiative (examples):
  - Wi‑Fi: uptime, session success, throughput
  - Boarding: boarding time variance, scan flow compliance, D+0 readiness
  - Crew: QA audit scores, complaint ratio, training completion
  - Seat: defect rates, seat-related maintenance events, cabin comfort survey
  - IFE: system health, fault rate, content update cadence, usage rate

### C) Guardrails (to avoid misleading conclusions)
- Review volume threshold for trend interpretation (e.g., total_reviews ≥ 50)
- Topic tagging coverage (% reviews with ≥1 topic)
- Model version stability check (probability drift; periodic evaluation)

---

## 3) Core metrics (formal definitions)

### 3.1 Topic mention rate
`topic_mention_rate[t] = topic_mentions[t] / total_reviews`

### 3.2 Negative mention rate (primary “frequency” proxy)
`topic_negative_mention_rate[t] = topic_neg_mentions[t] / total_reviews`

Where:
- `topic_neg_mentions[t] = count(reviews where topic[t]=1 and sentiment_pred='neg')`

### 3.3 Severity (intensity proxy)
`topic_severity[t] = mean(p_neg | topic[t]=1)`

### 3.4 Priority score (recent-window ranking)
`priority_score[t] = frequency_share_of_negative[t] × severity_avg_p_neg[t]`

Where:
- `frequency_share_of_negative[t] = topic_neg_mentions[t] / total_negative_reviews`
- `severity_avg_p_neg[t] = mean(p_neg | topic[t]=1 and sentiment_pred='neg')`

### 3.5 Spike detection (alert metric)
A simple baseline:
- Compute rolling mean and rolling std for `topic_negative_mention_rate[t]`
- Trigger if current value > mean + 2×std, and volume ≥ threshold

Optional upgrade:
- STL decomposition or robust z-score on residuals

---

## 4) Information architecture (dashboard pages)

### Page 1 — Executive Overview (high-level)
**Goal:** in <60 seconds, understand overall health and what changed.

**Widgets**
- KPI tiles:
  - Total reviews (recent period)
  - Overall negative rate
  - Topic tagging coverage
  - # active alerts
- “Top drivers of dissatisfaction (recent window)”:
  - Ranked bar: priority score (top 5–10)
- “What is getting worse?”:
  - Top 3 topics by Δ negative mention rate (MoM / QoQ)
- “What is improving?”:
  - Top 3 topics with best improvement (declines)
- Initiative outcomes snapshot:
  - % initiatives improving / flat / worsening

**Filters (global)**
- Time window: last 4 weeks / 12 weeks / 12 months / custom
- Segment: route / station / fleet / cabin (if available)
- Review channel / language (if available)

---

### Page 2 — Trend Explorer (frequency + severity)
**Goal:** diagnose whether dissatisfaction is rising due to more mentions or stronger negativity.

**Views**
- Small multiples: each topic shows two lines:
  - negative mention rate (rolling)
  - severity (avg p_neg)
- Optional “stacked contribution”:
  - contribution of topics to overall negative volume

**Controls**
- Volume filter (exclude low-sample months)
- Rolling window selector (4w, 8w, 3m)
- Topic selector (top N / manual)

---

### Page 3 — Priority Matrix (severity vs frequency)
**Goal:** prioritize themes for backlog planning.

**Plot**
- Scatter:
  - x = frequency share among negative
  - y = severity (avg p_neg)
  - bubble size = topic_neg_mentions (volume)
  - bubble color = Δ severity or Δ frequency

**Interactions**
- Click a topic to open the Topic Deep Dive page
- Hover shows topic stats + trend deltas + top keywords

---

### Page 4 — Topic Deep Dive (root-cause orientation)
**Goal:** move from “what” to “why” and identify the right owner.

**Sections**
- Topic summary card:
  - current negative mention rate, severity, volume
  - MoM/QoQ deltas
  - confidence flag (volume and stability)
- Segmentation table (top breakdowns):
  - by route / station / fleet / cabin / travel class
- Example review snippets:
  - top 10 reviews by p_neg (topic matched)
  - representative “median” negative examples
- Keyword / subtheme breakdown:
  - within-topic keyword histogram (optional)
  - NMF subtopic mapping (optional)

---

### Page 5 — Hotspots & Operations Map (advanced)
**Goal:** identify where to run pilots.

**Views**
- Heatmap:
  - rows = station/airport (or route)
  - cols = topics
  - metric = negative mention rate or severity
- “Hotspot list”:
  - top 10 segments with biggest deterioration

**Notes**
If route/fleet/station are not available, this page is included as an extension and can be mocked with placeholders in the portfolio narrative.

---

### Page 6 — Initiative Scorecards (closed-loop measurement)
**Goal:** track whether a specific initiative is working.

**Per-initiative scorecard**
- Header:
  - owner, start date, scope (routes/fleet), status
- VoC KPIs:
  - topic negative mention rate (pre vs post, rolling)
  - severity (pre vs post)
- Operational KPI(s):
  - telemetry / ops metric trend (pre vs post)
- Guardrails:
  - overall review volume, overall negative rate, adjacent topics

**Evaluation patterns**
- Before/after with matched scope (e.g., only affected fleet/route)
- Difference-in-differences (if a comparable control group exists)

---

### Page 7 — Alert Center (triage workflow)
**Goal:** surface spikes early and support action.

**Alert list**
- topic, segment, current value, baseline, z-score, volume
- “investigation checklist”:
  - confirm volume
  - confirm taxonomy match quality
  - identify top segments
  - review representative examples

**Auto-generated context**
- 5 most negative examples (highest p_neg)
- top co-occurring topics
- last known relevant operational event (optional)

---

## 5) Data model (semantic layer)

### Recommended tables
- `fact_reviews`
  - review_id, date, year_month, text_clean, rating, sentiment_pred, p_neg, p_neu, p_pos
- `dim_topic`
  - topic_id, topic_name, owner_team, keywords_version, description
- `bridge_review_topic`
  - review_id, topic_id, is_mentioned (0/1)
- `fact_topic_monthly`
  - year_month, topic_id, total_reviews, topic_mentions, topic_neg_mentions, neg_mention_rate, severity
- `dim_segment` (optional)
  - route, station, fleet, cabin, region
- `fact_initiative`
  - initiative_id, topic_id, start_date, end_date, owner, scope_segment
- `fact_alerts`
  - alert_id, date, topic_id, segment_id, metric, baseline, score, status

### Versioning
- `model_version`: sentiment model ID + training date
- `taxonomy_version`: keyword list version + change log

---

## 6) Data pipeline and operations (portfolio-level)

### ETL steps (daily/weekly)
1. Ingest reviews (raw → staged)
2. Clean text (normalize, remove noise)
3. Score sentiment (predict + probabilities)
4. Tag topics (taxonomy)
5. Aggregate monthly/weekly tables
6. Compute alerts (spike detection)
7. Publish to BI layer / dashboard

### Monitoring
- Data freshness SLA
- Volume anomalies
- Coverage drift (topic tagging)
- Model probability drift (distribution of p_neg)

---

## 7) Governance and quality controls

### Topic taxonomy QA (monthly)
- For each top topic:
  - sample 10–20 matched reviews
  - record false positives/negatives
  - update keywords
  - log changes (changelog)

### Model QA (quarterly)
- Recompute evaluation on a stable holdout
- Optional small human-labeled set for calibration
- Track neutral confusion patterns

### Interpretability
- Maintain a “metric dictionary” page in the dashboard
- Provide an “examples” panel to ground quantitative signals in text

---

## 8) High-level UI layout (wireframe)

### Executive Overview (example)
- Row 1: KPI tiles (Volume, Neg Rate, Coverage, Alerts)
- Row 2: Priority ranking bar + “Top worsening” list
- Row 3: Initiative outcomes (traffic-light summary)
- Row 4: Links to deep dives (topics and alerts)

### Topic Deep Dive (example)
- Left: trend lines (neg rate + severity)
- Right: segmentation table + hotspot list
- Bottom: representative review examples + keyword breakdown

---

## 9) Portfolio framing (how to describe this work)

This dashboard demonstrates:
- Building a **repeatable VoC measurement system**
- Turning unstructured text into **prioritized, trackable initiatives**
- Designing **metrics, governance, and operations** (not just a one-off analysis)
- Communicating insights in an executive-ready format

---

## 10) Implementation options (tool-agnostic)
- BI tools: Looker / Power BI / Tableau with a semantic layer
- Web app: Streamlit + DuckDB/Postgres + scheduled ETL
- Production: warehouse + dbt + monitoring + access control

This document is intentionally tool-agnostic so it remains valid across stacks.
