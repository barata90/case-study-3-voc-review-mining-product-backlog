import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ----------------------------
# Config
# ----------------------------
RANDOM_STATE = 42
DATA_PATH_DEFAULT = "airline_reviews_skytrax.csv"

st.set_page_config(
    page_title="VoC Dashboard — Review Mining → Backlog",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------
# Topic taxonomy (interpretable, editable)
# ----------------------------
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "seat_comfort": [
        r"\bseat\b", r"\bseats\b", r"\bseating\b",
        r"leg\s*room", r"\blegroom\b", r"\brecline\b",
        r"\bcushion\w*\b", r"\bcomfort\b", r"\buncomfortable\b",
    ],
    "crew_service": [
        r"\bcrew\b", r"\bcabin\s*staff\b", r"\bflight\s*attendant\w*\b",
        r"\bservice\b", r"\battitude\b", r"\bfriendly\b", r"\brude\b",
        r"\bhelpful\b", r"\bprofessional\b",
    ],
    "food_bev": [
        r"\bfood\b", r"\bmeal\w*\b", r"\bbreakfast\b", r"\blunch\b", r"\bdinner\b",
        r"\bsnack\w*\b", r"\bdrink\w*\b", r"\bbeverage\w*\b",
    ],
    "wifi": [
        r"\bwifi\b", r"\bwi\-?fi\b", r"\binternet\b",
        r"\bconnect\w*\b", r"\bnetwork\b",
    ],
    "boarding_gate": [
        r"\bboarding\b", r"\bgate\b", r"\bqueue\b",
        r"priority\s*boarding", r"\bzone\b", r"\bboarding\s*pass\b",
    ],
    "delay_punctuality": [
        r"\bdelay\w*\b", r"\blate\b", r"\bcancel\w*\b", r"\bdivert\w*\b",
        r"\bmissed\s*connection\b", r"\birregular\s*ops\b", r"\bdisruption\b",
    ],
    "baggage": [
        r"\bbaggage\b", r"\bluggage\b", r"\bbag\w*\b",
        r"\bcarousel\b", r"\blost\b", r"\bdamage\w*\b",
        r"\bclaim\b",
    ],
    "cleanliness": [
        r"\bclean\b", r"\bdirty\b", r"\btoilet\b", r"\blavatory\b",
        r"\bhygiene\b", r"\bsmell\w*\b",
    ],
    "digital_booking": [
        r"\bbooking\b", r"\bwebsite\b", r"\bapp\b",
        r"online\s*check\-?in", r"\bcheck\s*in\b",
        r"\brefund\w*\b", r"\bcustomer\s*support\b", r"\bcontact\s*center\b",
        r"\bpayment\b", r"\bfees?\b",
    ],
    "entertainment": [
        r"\binflight\s*entertain\w*\b", r"\bentertain\w*\b",
        r"\bmovie\w*\b", r"\bmusic\b", r"\bscreen\b", r"\bife\b",
        r"\bheadphone\w*\b",
    ],
}

TOPIC_REGEX = {k: re.compile("|".join(v), flags=re.IGNORECASE) for k, v in TOPIC_KEYWORDS.items()}


# ----------------------------
# Helpers
# ----------------------------
def clean_text(s: str) -> str:
    """Lightweight normalization suitable for TF-IDF baselines."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip().lower()
    # keep letters/numbers/basic punctuation for n-grams; remove excessive symbols
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\/']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rating_to_sentiment(r: float) -> Optional[str]:
    if pd.isna(r):
        return np.nan
    try:
        r = float(r)
    except Exception:
        return np.nan
    if r >= 8:
        return "pos"
    if r <= 4:
        return "neg"
    return "neu"


def tag_topics(text: str) -> Dict[str, int]:
    text = "" if text is None else str(text)
    return {topic: int(bool(rx.search(text))) for topic, rx in TOPIC_REGEX.items()}


def rolling_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


# ----------------------------
# Data + model pipeline (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse date and derive year_month
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["text_clean"] = df["content"].astype(str).map(clean_text)
    return df


@st.cache_data(show_spinner=False)
def build_weak_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["sentiment_label"] = df["overall_rating"].map(rating_to_sentiment)
    labeled = df.dropna(subset=["sentiment_label"]).copy()
    return df, labeled


@st.cache_resource(show_spinner=False)
def train_sentiment_model(labeled: pd.DataFrame) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        labeled["text_clean"],
        labeled["sentiment_label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labeled["sentiment_label"],
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_features=40000,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=300,
        n_jobs=-1,
        # multinomial is default in modern sklearn; avoid deprecated param
    )
    clf.fit(Xtr, y_train)

    pred = clf.predict(Xte)
    report = classification_report(y_test, pred, digits=3, output_dict=False)
    cm = confusion_matrix(y_test, pred, labels=["neg", "neu", "pos"])

    metrics = {"report_text": report, "confusion_matrix": cm, "labels": ["neg", "neu", "pos"]}
    return vectorizer, clf, metrics


@st.cache_data(show_spinner=False)
def score_sentiment(df: pd.DataFrame, _vectorizer: TfidfVectorizer, _clf: LogisticRegression) -> pd.DataFrame:
    df = df.copy()
    X_all = _vectorizer.transform(df["text_clean"])
    proba = _clf.predict_proba(X_all)
    classes = _clf.classes_.tolist()
    proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])

    df = pd.concat([df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    df["sentiment_pred"] = _clf.predict(X_all)

    df["p_pos"] = df.get("p_pos", 0.0)
    df["p_neg"] = df.get("p_neg", 0.0)
    df["sentiment_score"] = (df["p_pos"].astype(float) - df["p_neg"].astype(float))
    df["is_negative"] = (df["sentiment_pred"] == "neg").astype(int)
    return df


@st.cache_data(show_spinner=False)
def add_topic_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flags = df["content"].astype(str).map(tag_topics)
    topic_df = pd.DataFrame(list(flags))
    df = pd.concat([df.reset_index(drop=True), topic_df.reset_index(drop=True)], axis=1)
    return df


@st.cache_data(show_spinner=False)
def build_monthly_trend(df: pd.DataFrame, topics: List[str]) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("year_month"):
        total = len(g)
        row = {"year_month": m, "total_reviews": total}
        for t in topics:
            topic_mask = g[t] == 1
            neg_mask = (g["is_negative"] == 1) & topic_mask
            row[f"{t}_neg_count"] = int(neg_mask.sum())
            row[f"{t}_neg_rate"] = float(neg_mask.sum() / total) if total else 0.0
            if topic_mask.any() and "p_neg" in g.columns:
                row[f"{t}_severity"] = float(g.loc[topic_mask, "p_neg"].mean())
            else:
                row[f"{t}_severity"] = np.nan
        rows.append(row)

    trend = pd.DataFrame(rows).sort_values("year_month")
    trend["year_month_dt"] = pd.to_datetime(trend["year_month"] + "-01", errors="coerce")
    trend = trend.dropna(subset=["year_month_dt"]).sort_values("year_month_dt")
    return trend


@st.cache_data(show_spinner=False)
def compute_recent_priority(df: pd.DataFrame, trend: pd.DataFrame, topics: List[str], last_n_months: int) -> pd.DataFrame:
    # Use the last N months observed in the trend table
    months = trend["year_month"].unique()
    if len(months) == 0:
        return pd.DataFrame(columns=["topic", "neg_mentions", "frequency_share_of_negative", "severity_avg_p_neg", "priority_score"])
    recent_months = months[-last_n_months:]
    df_recent = df[df["year_month"].isin(recent_months)].copy()

    neg_total = int((df_recent["is_negative"] == 1).sum())
    rows = []
    for t in topics:
        topic_mask = df_recent[t] == 1
        neg_topic = df_recent[(df_recent["is_negative"] == 1) & topic_mask]
        freq = (len(neg_topic) / neg_total) if neg_total else 0.0
        sev = float(neg_topic["p_neg"].mean()) if (len(neg_topic) and "p_neg" in df_recent.columns) else np.nan
        rows.append(
            {
                "topic": t,
                "neg_mentions": int(len(neg_topic)),
                "frequency_share_of_negative": float(freq),
                "severity_avg_p_neg": float(sev) if not np.isnan(sev) else np.nan,
                "priority_score": float(freq * (sev if not np.isnan(sev) else 0.0)),
            }
        )

    prio = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    return prio


@st.cache_data(show_spinner=False)
def compute_alerts(trend: pd.DataFrame, topics: List[str], window: int, z_thresh: float, min_reviews: int) -> pd.DataFrame:
    """Alert if last point exceeds rolling mean + z * rolling std (with volume threshold)."""
    if trend.empty:
        return pd.DataFrame()

    last_row = trend.iloc[-1]
    alerts = []
    for t in topics:
        s = trend[f"{t}_neg_rate"].astype(float)
        # rolling baseline excluding current point for stability
        baseline = s.shift(1).rolling(window, min_periods=max(3, window // 2))
        mu = baseline.mean()
        sd = baseline.std(ddof=0)
        cur = float(last_row[f"{t}_neg_rate"])
        cur_vol = int(last_row["total_reviews"])

        mu_last = float(mu.iloc[-1]) if not np.isnan(mu.iloc[-1]) else np.nan
        sd_last = float(sd.iloc[-1]) if not np.isnan(sd.iloc[-1]) else np.nan

        if cur_vol < min_reviews or np.isnan(mu_last) or np.isnan(sd_last) or sd_last == 0:
            status = "insufficient_data"
            score = np.nan
            is_alert = False
        else:
            score = (cur - mu_last) / sd_last
            is_alert = bool(score >= z_thresh)
            status = "alert" if is_alert else "ok"

        alerts.append(
            {
                "topic": t,
                "period": str(last_row["year_month"]),
                "current_neg_rate": cur,
                "baseline_mean": mu_last,
                "baseline_std": sd_last,
                "z_score": score,
                "total_reviews": cur_vol,
                "status": status,
            }
        )

    alert_df = pd.DataFrame(alerts).sort_values(["status", "z_score"], ascending=[True, False])
    return alert_df


def format_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100*x:.1f}%"


# ----------------------------
# Load & build pipeline
# ----------------------------
st.title("Voice of Customer Dashboard — Review Mining → Product Backlog")

with st.sidebar:
    st.header("Controls")
    csv_path = st.text_input("Dataset path (CSV)", value=DATA_PATH_DEFAULT)
    last_n_months = st.slider("Recent window (months)", min_value=3, max_value=24, value=12, step=1)
    min_reviews = st.slider("Min reviews per month (trend/alerts)", min_value=10, max_value=200, value=50, step=10)
    rolling_w = st.slider("Rolling window (months) for charts", min_value=1, max_value=6, value=3, step=1)
    alert_window = st.slider("Alert baseline window (months)", min_value=3, max_value=12, value=6, step=1)
    z_thresh = st.slider("Alert threshold (z-score)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)

    st.divider()
    st.caption("This mock dashboard recomputes sentiment and topics from the raw dataset and caches results for responsiveness.")

# Load data
try:
    df_raw = load_data(csv_path)
except Exception as e:
    st.error(f"Failed to load CSV at '{csv_path}'. Error: {e}")
    st.stop()

df_raw, labeled = build_weak_labels(df_raw)

# Train model (cached)
with st.spinner("Training / loading sentiment model..."):
    vec, clf, metrics = train_sentiment_model(labeled)

# Score sentiment
with st.spinner("Scoring sentiment for all reviews..."):
    df_scored = score_sentiment(df_raw, vec, clf)

# Tag topics
with st.spinner("Tagging topics (aspect taxonomy)..."):
    df_scored = add_topic_flags(df_scored)

topics = list(TOPIC_KEYWORDS.keys())

# Trend + priority + alerts
trend = build_monthly_trend(df_scored, topics)
prio = compute_recent_priority(df_scored, trend, topics, last_n_months)
alerts = compute_alerts(trend[trend["total_reviews"] >= min_reviews], topics, alert_window, z_thresh, min_reviews)

# Recent window slice
months_all = trend["year_month"].unique()
recent_months = months_all[-last_n_months:] if len(months_all) else []
df_recent = df_scored[df_scored["year_month"].isin(recent_months)].copy()

# Global health KPIs
recent_total = len(df_recent)
recent_neg_rate = float(df_recent["is_negative"].mean()) if recent_total else np.nan
topic_any = df_recent[topics].sum(axis=1) > 0
topic_coverage = float(topic_any.mean()) if recent_total else np.nan
n_alerts = int((alerts["status"] == "alert").sum()) if not alerts.empty else 0


# ----------------------------
# Tabs (aligned to advanced spec)
# ----------------------------
tabs = st.tabs([
    "Executive Overview",
    "Trend Explorer",
    "Priority Matrix",
    "Topic Deep Dive",
    "Hotspots",
    "Initiative Scorecards",
    "Alert Center",
])

# ----------------------------
# Tab 1: Executive Overview
# ----------------------------
with tabs[0]:
    st.subheader("Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews (recent window)", f"{recent_total:,}")
    c2.metric("Overall negative rate", format_pct(recent_neg_rate))
    c3.metric("Topic tagging coverage", format_pct(topic_coverage))
    c4.metric("Active alerts", f"{n_alerts:,}")

    st.markdown("### Top drivers of dissatisfaction (recent window)")
    if prio.empty:
        st.info("Priority table is empty. Confirm that the dataset contains valid dates and review text.")
    else:
        top_n = st.slider("Number of topics to display", min_value=5, max_value=len(prio), value=min(10, len(prio)), step=1, key="top_n_exec")
        top = prio.head(top_n).iloc[::-1].copy()  # reverse for barh
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["topic"], top["priority_score"])
        ax.set_xlabel("Priority score (frequency × severity)")
        ax.set_ylabel("Topic")
        ax.set_title("Top topics by priority score")
        st.pyplot(fig, clear_figure=True)

    st.markdown("### What is changing (trend deltas)")
    # Compute simple deltas: last 3 months vs previous 3 months (within filtered trend)
    trend_f = trend[trend["total_reviews"] >= min_reviews].copy()
    if len(trend_f) >= 6:
        last3 = trend_f.tail(3)
        prev3 = trend_f.tail(6).head(3)
        delta_rows = []
        for t in topics:
            d_neg = float(last3[f"{t}_neg_rate"].mean() - prev3[f"{t}_neg_rate"].mean())
            d_sev = float(last3[f"{t}_severity"].mean(skipna=True) - prev3[f"{t}_severity"].mean(skipna=True))
            delta_rows.append({"topic": t, "delta_neg_rate": d_neg, "delta_severity": d_sev})
        deltas = pd.DataFrame(delta_rows)
        worsening = deltas.sort_values("delta_neg_rate", ascending=False).head(5)
        improving = deltas.sort_values("delta_neg_rate", ascending=True).head(5)

        lcol, rcol = st.columns(2)
        with lcol:
            st.caption("Top topics getting worse (negative mention rate)")
            st.dataframe(worsening.style.format({"delta_neg_rate": "{:.3f}", "delta_severity": "{:.3f}"}), use_container_width=True)
        with rcol:
            st.caption("Top topics improving (negative mention rate)")
            st.dataframe(improving.style.format({"delta_neg_rate": "{:.3f}", "delta_severity": "{:.3f}"}), use_container_width=True)
    else:
        st.info("Not enough high-volume months to compute 3-month deltas. Consider lowering the minimum review threshold in the sidebar.")

    st.markdown("### Model evaluation (baseline)")
    with st.expander("Show classification report and confusion matrix"):
        st.text(metrics["report_text"])
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=metrics["labels"], columns=metrics["labels"])
        st.dataframe(cm_df, use_container_width=True)

# ----------------------------
# Tab 2: Trend Explorer
# ----------------------------
with tabs[1]:
    st.subheader("Trend Explorer (frequency + severity)")

    trend_f = trend[trend["total_reviews"] >= min_reviews].copy()
    if trend_f.empty:
        st.info("No months meet the minimum review threshold. Lower the threshold to view trends.")
    else:
        # Default to top 5 topics by priority
        default_topics = prio.head(5)["topic"].tolist() if not prio.empty else topics[:5]
        selected = st.multiselect("Topics", options=topics, default=default_topics)

        metric_choice = st.radio("Metric", ["Negative mention rate", "Severity (avg p_neg)"], horizontal=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        for t in selected:
            if metric_choice == "Negative mention rate":
                y = rolling_mean(trend_f[f"{t}_neg_rate"], rolling_w)
                ax.plot(trend_f["year_month_dt"], y, label=t)
            else:
                y = rolling_mean(trend_f[f"{t}_severity"].astype(float), rolling_w)
                ax.plot(trend_f["year_month_dt"], y, label=t)

        ax.set_title(f"Monthly trend — {metric_choice} (rolling {rolling_w} months)")
        ax.set_xlabel("Month")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.legend(ncols=2, fontsize=9)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, clear_figure=True)

        st.caption("Tip: Severity helps distinguish whether a rising trend is driven by more mentions or more intense dissatisfaction.")

# ----------------------------
# Tab 3: Priority Matrix
# ----------------------------
with tabs[2]:
    st.subheader("Priority Matrix (severity vs frequency) — recent window")

    if prio.empty:
        st.info("Priority table is empty.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = prio["frequency_share_of_negative"].astype(float)
        y = prio["severity_avg_p_neg"].astype(float)
        sizes = (prio["neg_mentions"].astype(float).clip(lower=1)) ** 0.5 * 20  # sqrt scaling

        ax.scatter(x, y, s=sizes)
        for _, r in prio.iterrows():
            ax.annotate(r["topic"], (r["frequency_share_of_negative"], r["severity_avg_p_neg"]), fontsize=8)

        ax.set_title("Issue prioritization — severity vs frequency (recent window)")
        ax.set_xlabel("Frequency share (among negative reviews)")
        ax.set_ylabel("Severity (avg p_neg)")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, clear_figure=True)

        st.markdown(
            """
**Interpretation guide**
- **Top-right:** highest priority (high frequency + high severity)  
- **Top-left:** sharp pain points (low frequency, high severity)  
- **Bottom-right:** widespread friction (high frequency, lower severity)  
- **Bottom-left:** watchlist
"""
        )

# ----------------------------
# Tab 4: Topic Deep Dive
# ----------------------------
with tabs[3]:
    st.subheader("Topic Deep Dive")

    if prio.empty:
        topic_sel = st.selectbox("Topic", options=topics, index=0)
    else:
        topic_sel = st.selectbox("Topic", options=topics, index=int(prio["topic"].tolist().index(prio.iloc[0]["topic"])) if prio.iloc[0]["topic"] in topics else 0)

    # Summary stats (recent window)
    topic_mask = df_recent[topic_sel] == 1
    topic_mentions = int(topic_mask.sum())
    topic_neg_mentions = int(((df_recent["is_negative"] == 1) & topic_mask).sum())
    topic_neg_rate = float(topic_neg_mentions / recent_total) if recent_total else np.nan
    topic_sev = float(df_recent.loc[topic_mask, "p_neg"].mean()) if topic_mentions else np.nan

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Topic mentions (recent)", f"{topic_mentions:,}")
    a2.metric("Negative mentions (recent)", f"{topic_neg_mentions:,}")
    a3.metric("Negative mention rate", format_pct(topic_neg_rate))
    a4.metric("Severity (avg p_neg)", "—" if np.isnan(topic_sev) else f"{topic_sev:.3f}")

# Trend for the topic
trend_f = trend[trend["total_reviews"] >= min_reviews].copy()
if not trend_f.empty:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        trend_f["year_month_dt"],
        rolling_mean(trend_f[f"{topic_sel}_neg_rate"], rolling_w),
        label="Negative mention rate"
    )

    ax2 = ax.twinx()
    ax2.plot(
        trend_f["year_month_dt"],
        rolling_mean(trend_f[f"{topic_sel}_severity"].astype(float), rolling_w),
        linestyle="--",
        label="Severity (avg p_neg)"
    )

    ax.set_title(f"Topic trend — {topic_sel} (rolling {rolling_w} months)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Negative mention rate")
    ax2.set_ylabel("Severity")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ✅ Robust diagonal ticks in Streamlit
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Segment breakdowns (recent window)")
    seg_cols = ["airline_name", "route", "cabin_flown", "aircraft", "author_country"]
    seg_cols = [c for c in seg_cols if c in df_recent.columns]

    seg_choice = st.selectbox("Segment dimension", options=seg_cols, index=0)
    seg_df = df_recent[topic_mask].copy()

    if seg_df.empty:
        st.info("No mentions found for the selected topic in the recent window.")
    else:
        agg = (
            seg_df.groupby(seg_choice)
            .agg(
                mentions=("content", "size"),
                neg_mentions=("is_negative", "sum"),
                severity=("p_neg", "mean"),
            )
            .sort_values(["neg_mentions", "mentions"], ascending=False)
            .head(15)
            .reset_index()
        )
        agg["neg_rate_within_segment_mentions"] = agg["neg_mentions"] / agg["mentions"].replace(0, np.nan)
        st.dataframe(
            agg.style.format({"severity": "{:.3f}", "neg_rate_within_segment_mentions": "{:.1%}"}),
            use_container_width=True,
        )

    st.markdown("### Representative review examples")
    ex = df_recent[topic_mask].copy()
    if ex.empty:
        st.info("No examples available for this topic in the selected window.")
    else:
        ex = ex.sort_values("p_neg", ascending=False).head(10)
        show_cols = [c for c in ["date", "airline_name", "route", "cabin_flown", "p_neg", "sentiment_pred", "content"] if c in ex.columns]
        ex = ex[show_cols].copy()
        # Truncate long text for readability
        if "content" in ex.columns:
            ex["content"] = ex["content"].astype(str).str.slice(0, 220) + "…"
        st.dataframe(ex, use_container_width=True)

    st.markdown("### Topic co-occurrence (among topic mentions)")
    if topic_mentions:
        co = {}
        base = df_recent[topic_mask]
        for t in topics:
            if t == topic_sel:
                continue
            co[t] = float((base[t] == 1).mean())
        co_df = pd.DataFrame({"topic": list(co.keys()), "co_occurrence_rate": list(co.values())}).sort_values("co_occurrence_rate", ascending=False).head(10)
        st.dataframe(co_df.style.format({"co_occurrence_rate": "{:.1%}"}), use_container_width=True)

# ----------------------------
# Tab 5: Hotspots
# ----------------------------
with tabs[4]:
    st.subheader("Hotspots (advanced mock)")

    st.caption("Hotspots are computed from available fields (airline, route, cabin, aircraft). Heatmaps highlight where negative mentions concentrate.")

    focus_topics = prio.head(5)["topic"].tolist() if not prio.empty else topics[:5]
    dim = st.selectbox("Hotspot dimension", options=[c for c in ["airline_name", "route", "cabin_flown", "aircraft", "author_country"] if c in df_recent.columns], index=0)

    # Compute per-segment negative mention rate (within all reviews for the segment)
    rows = []
    for seg, g in df_recent.groupby(dim):
        total = len(g)
        if total < 30:
            continue
        row = {"segment": seg, "total_reviews": total}
        for t in focus_topics:
            topic_neg = int(((g["is_negative"] == 1) & (g[t] == 1)).sum())
            row[f"{t}_neg_rate"] = topic_neg / total
        rows.append(row)

    if not rows:
        st.info("Not enough volume to compute hotspots. Try a different segment dimension or broaden the recent window.")
    else:
        heat = pd.DataFrame(rows).sort_values("total_reviews", ascending=False).head(20).reset_index(drop=True)
        st.dataframe(heat.style.format({c: "{:.1%}" for c in heat.columns if c.endswith("_neg_rate")}), use_container_width=True)

        # Heatmap visualization (matplotlib)
        mat_cols = [c for c in heat.columns if c.endswith("_neg_rate")]
        mat = heat[mat_cols].to_numpy()
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mat, aspect="auto")
        ax.set_yticks(np.arange(len(heat)))
        ax.set_yticklabels([str(s)[:28] for s in heat["segment"].tolist()])
        ax.set_xticks(np.arange(len(mat_cols)))
        ax.set_xticklabels([c.replace("_neg_rate", "") for c in mat_cols], rotation=45, ha="right")
        ax.set_title(f"Hotspot heatmap — negative mention rate by {dim} (recent window)")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        st.pyplot(fig, clear_figure=True)

# ----------------------------
# Tab 6: Initiative Scorecards
# ----------------------------
with tabs[5]:
    st.subheader("Initiative Scorecards (closed-loop measurement mock)")

    # Load focus initiatives if present; otherwise generate a minimal table from top topics
    initiatives_path = "backlog_focus_top5.csv"
    try:
        init_df = pd.read_csv(initiatives_path)
    except Exception:
        # Minimal fallback
        init_df = pd.DataFrame(
            {
                "initiative": [f"Improve {t.replace('_', ' ')}" for t in (prio.head(5)["topic"].tolist() if not prio.empty else topics[:5])],
                "primary_topic": (prio.head(5)["topic"].tolist() if not prio.empty else topics[:5]),
                "kpi_success": ["Topic negative mention rate ↓; topic severity ↓"] * 5,
                "how_to_track": ["VoC trend + relevant ops telemetry"] * 5,
            }
        )

    init_sel = st.selectbox("Initiative", options=init_df["initiative"].tolist(), index=0)
    sel = init_df[init_df["initiative"] == init_sel].iloc[0]
    t = sel.get("primary_topic", None)

    if t not in topics:
        st.warning("Selected initiative does not map to a known topic in the taxonomy.")
    else:
        st.markdown(f"**Primary topic:** `{t}`")
        st.write(f"**Success KPI(s):** {sel.get('kpi_success', '')}")
        st.write(f"**Tracking plan:** {sel.get('how_to_track', '')}")

        trend_f = trend[trend["total_reviews"] >= min_reviews].copy()
        if len(trend_f) < 8:
            st.info("Not enough high-volume months to show a meaningful pre/post comparison.")
        else:
            # Mock start month selector
            start_idx = st.slider("Initiative start month (index on trend)", min_value=0, max_value=len(trend_f)-1, value=max(0, len(trend_f)-6), step=1)
            start_dt = trend_f.iloc[start_idx]["year_month_dt"]

            pre = trend_f[trend_f["year_month_dt"] < start_dt].tail(3)
            post = trend_f[trend_f["year_month_dt"] >= start_dt].head(3)

            pre_rate = float(pre[f"{t}_neg_rate"].mean()) if len(pre) else np.nan
            post_rate = float(post[f"{t}_neg_rate"].mean()) if len(post) else np.nan
            pre_sev = float(pre[f"{t}_severity"].mean(skipna=True)) if len(pre) else np.nan
            post_sev = float(post[f"{t}_severity"].mean(skipna=True)) if len(post) else np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pre (3m) neg rate", format_pct(pre_rate))
            m2.metric("Post (3m) neg rate", format_pct(post_rate), delta=("—" if np.isnan(pre_rate) or np.isnan(post_rate) else f"{(post_rate-pre_rate)*100:+.1f} pp"))
            m3.metric("Pre (3m) severity", "—" if np.isnan(pre_sev) else f"{pre_sev:.3f}")
            m4.metric("Post (3m) severity", "—" if np.isnan(post_sev) else f"{post_sev:.3f}", delta=("—" if np.isnan(pre_sev) or np.isnan(post_sev) else f"{(post_sev-pre_sev):+.3f}"))

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(trend_f["year_month_dt"], rolling_mean(trend_f[f"{t}_neg_rate"], rolling_w), label="Neg mention rate")
            ax.axvline(start_dt, linestyle="--")
            ax.set_title(f"Initiative scorecard trend — {t}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Negative mention rate")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.xticks(rotation=45)
            ax.grid(True, alpha=0.2)
            st.pyplot(fig, clear_figure=True)

# ----------------------------
# Tab 7: Alert Center
# ----------------------------
with tabs[6]:
    st.subheader("Alert Center")

    st.caption("Alerts use a rolling baseline on topic negative mention rate and trigger when the latest period exceeds the baseline by a z-score threshold.")

    if alerts.empty:
        st.info("No alerts computed (insufficient high-volume months or baseline window too strict).")
    else:
        st.dataframe(
            alerts.style.format(
                {
                    "current_neg_rate": "{:.2%}",
                    "baseline_mean": "{:.2%}",
                    "baseline_std": "{:.3f}",
                    "z_score": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        alert_topics = alerts[alerts["status"] == "alert"]["topic"].tolist()
        if not alert_topics:
            st.success("No active alerts at the current threshold.")
        else:
            st.markdown("### Investigate an alert")
            t = st.selectbox("Alert topic", options=alert_topics)
            # Show representative examples for this topic in the latest period
            latest_period = str(trend.iloc[-1]["year_month"]) if not trend.empty else None
            ex = df_scored[(df_scored["year_month"] == latest_period) & (df_scored[t] == 1)].copy()
            if ex.empty:
                st.info("No matching reviews found for the alert period and topic.")
            else:
                ex = ex.sort_values("p_neg", ascending=False).head(10)
                show_cols = [c for c in ["date", "airline_name", "route", "cabin_flown", "p_neg", "sentiment_pred", "content"] if c in ex.columns]
                ex = ex[show_cols].copy()
                if "content" in ex.columns:
                    ex["content"] = ex["content"].astype(str).str.slice(0, 220) + "…"
                st.dataframe(ex, use_container_width=True)

                # Co-occurring topics in alert set
                base = df_scored[(df_scored["year_month"] == latest_period) & (df_scored[t] == 1)]
                co = {u: float((base[u] == 1).mean()) for u in topics if u != t}
                co_df = pd.DataFrame({"topic": list(co.keys()), "co_occurrence_rate": list(co.values())}).sort_values("co_occurrence_rate", ascending=False).head(10)
                st.caption("Top co-occurring topics (alert set)")
                st.dataframe(co_df.style.format({"co_occurrence_rate": "{:.1%}"}), use_container_width=True)

st.caption("Portfolio note: This app is intentionally tool-agnostic and focuses on analytics logic, metric definitions, and cross-functional translation into initiatives.")
