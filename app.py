"""
CredCheck — Streamlit app (lean 13-feature model).
Run locally:  streamlit run app.py
Deploy:       push app.py + credcheck_model.py + requirements.txt to GitHub,
              then share.streamlit.io -> New app -> main file = app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from credcheck_model import build_default_model, prepare, GROUPS, FIRM_TYPES

st.set_page_config(page_title="Adaptive Trade-Credit Limits", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Training the model (first load only)…")
def get_model():
    model, _ = build_default_model()
    return model


model = get_model()
TOGGLE_GROUPS = ["trade", "gst", "bank", "bureau", "fin"]      # firmographics always present
GROUP_LABEL = {"trade": "Trade history with anchor", "gst": "GST / GSTN",
               "bank": "Bank statements (AA)", "bureau": "Credit bureau",
               "fin": "Financial statements"}


def risk_grade(p):
    return ("A (low)" if p < 0.05 else "B (moderate)" if p < 0.12
            else "C (elevated)" if p < 0.25 else "D (high)")


def reason_codes(row, scored):
    active = [g for g in GROUPS if g not in ("all", "firmo")
              and (1 - row[GROUPS[g]].isna().mean(axis=1).iloc[0]) >= 0.30]
    gated = [GROUP_LABEL[g] for g in TOGGLE_GROUPS if g not in active]
    notes = [f"✅ Scored using {len(active)} evidence source(s): "
             + ", ".join(GROUP_LABEL[g] for g in active)]
    if gated:
        notes.append("⚪ No data (expert switched off, **not** penalised): " + ", ".join(gated))
    if scored["confidence"].iloc[0] < 0.70:
        notes.append("🔎 Thin data → limit sized **conservatively** (lower confidence, not higher risk).")
    r = row.iloc[0]
    if pd.notna(r.get("bureau_score")):
        notes.append("➕ Strong bureau score." if r["bureau_score"] >= 750
                     else "➖ Weak bureau score." if r["bureau_score"] < 650 else "• Average bureau score.")
    if pd.notna(r.get("trade_ontime_ratio")):
        if r["trade_ontime_ratio"] < 0.70:
            notes.append("➖ Frequent late payments to the anchor.")
        elif r["trade_ontime_ratio"] >= 0.90:
            notes.append("➕ Excellent on-time payment record.")
    if pd.notna(r.get("fin_debt_to_equity")) and r["fin_debt_to_equity"] > 3:
        notes.append("➖ High leverage (debt-to-equity > 3).")
    if pd.notna(r.get("bank_bounce_rate")) and r["bank_bounce_rate"] > 0.10:
        notes.append("➖ Elevated bounce rate.")
    return notes


st.title("Adaptive Trade-Credit Limit Engine")
st.caption("The risk estimate does **not** punish missing data — "
           "absent evidence lowers *confidence* and sizes the limit conservatively instead.")

tab_single, tab_batch, tab_about = st.tabs(["Score one firm", "Score a CSV", "How it works"])

# -------------------------------------------------- SINGLE FIRM
with tab_single:
    st.markdown("#### Firm profile")
    c1, c2 = st.columns(2)
    firm_type = c1.selectbox("Legal structure", FIRM_TYPES, index=0)
    vintage = c2.number_input("Years in business", 0.2, 30.0, 5.0, 0.5)

    row = {"firm_type": firm_type, "firmo_vintage_years": vintage}
    for col in [c for cols in GROUPS.values() for c in cols if c not in row]:
        row[col] = np.nan

    st.markdown("#### Available evidence  ·  *toggle a source off to simulate missing data*")
    cols = st.columns(len(TOGGLE_GROUPS))
    have = {g: cols[i].toggle(GROUP_LABEL[g], value=(g != "fin")) for i, g in enumerate(TOGGLE_GROUPS)}

    with st.expander("Enter values for the available sources", expanded=True):
        if have["trade"]:
            st.markdown("**Trade history with the anchor**")
            a, b, c = st.columns(3)
            row["trade_ontime_ratio"] = a.slider("On-time payment ratio", 0.0, 1.0, 0.88)
            row["trade_days_to_pay_vs_terms"] = b.number_input("Avg days to pay vs terms", -20.0, 60.0, 2.0)
            row["trade_order_vol_monthly"] = c.number_input("Monthly order value (₹)", 0, 50_000_000, 300_000, step=10_000)
        if have["gst"]:
            st.markdown("**GST**")
            a, b = st.columns(2)
            row["gst_turnover_monthly"] = a.number_input("Monthly GST turnover (₹)", 0, 100_000_000, 500_000, step=10_000)
            row["gst_turnover_trend"] = b.slider("Turnover trend (YoY)", -0.3, 0.5, 0.08)
        if have["bank"]:
            st.markdown("**Bank statements**")
            a, b = st.columns(2)
            row["bank_avg_balance"] = a.number_input("Average balance (₹)", 0, 50_000_000, 150_000, step=10_000)
            row["bank_bounce_rate"] = b.slider("Cheque/auto-debit bounce rate", 0.0, 0.5, 0.02)
        if have["bureau"]:
            st.markdown("**Credit bureau**")
            a, b = st.columns(2)
            row["bureau_score"] = a.slider("Bureau score", 300, 900, 740)
            row["bureau_dpd_max_12m"] = b.number_input("Max days past due (12m)", 0, 180, 0)
        if have["fin"]:
            st.markdown("**Financial statements**")
            a, b = st.columns(2)
            row["fin_debt_to_equity"] = a.slider("Debt-to-equity", 0.05, 8.0, 1.0)
            row["fin_current_ratio"] = b.slider("Current ratio", 0.2, 5.0, 1.6)

    df_row = prepare(pd.DataFrame([row]))
    scored = model.score(df_row)
    pd_val = float(scored["pd"].iloc[0]); conf = float(scored["confidence"].iloc[0])
    limit = float(scored["recommended_limit"].iloc[0])

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended limit", f"₹{limit:,.0f}")
    m2.metric("Probability of default", f"{pd_val*100:.1f}%")
    m3.metric("Confidence", f"{conf*100:.0f}%")
    m4.metric("Risk grade", risk_grade(pd_val))
    st.progress(min(conf, 1.0), text=f"Data confidence: {conf*100:.0f}%")
    st.markdown("##### Reason codes")
    for n in reason_codes(df_row, scored):
        st.markdown("- " + n)

# -------------------------------------------------- BATCH
with tab_batch:
    st.markdown("Upload a CSV of firms (any of the standard columns; missing ones are fine). "
                "Try the sample files like `thinfile_proprietorships.csv`.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        out = model.score(prepare(raw))
        result = pd.concat([raw.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        st.dataframe(result, use_container_width=True, height=380)
        a, b, c = st.columns(3)
        a.metric("Firms scored", len(out))
        b.metric("Median limit", f"₹{out['recommended_limit'].median():,.0f}")
        c.metric("Median confidence", f"{out['confidence'].median()*100:.0f}%")
        st.download_button("Download scored CSV", result.to_csv(index=False), "scored.csv", "text/csv")

# -------------------------------------------------- ABOUT
with tab_about:
    st.markdown("""
**The principle.** Missing data is not bad data. A firm we know *less* about is scored with
*more uncertainty* → a more conservative limit, never a worse risk estimate.

**How.** One gradient-boosting *expert* per evidence source (trade, GST, bank, bureau,
financials) plus a generalist — 13 features in total. For each firm, only the experts whose
data is present are activated; their scores are blended by `coverage × reliability`,
calibrated to a real probability, and paired with a *confidence* that falls as data thins.
The limit engine then sizes a transparent line: `capacity × risk_multiplier × confidence`,
capped by affordability and concentration limits.

**Try it:** in *Score one firm*, switch off *Financial statements* and *Credit bureau* — the
probability of default barely moves, but confidence drops and the limit becomes more cautious.
""")
