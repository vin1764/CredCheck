"""
CredCheck — Streamlit app.
Run locally:   streamlit run app.py
Deploy:        push this + credcheck_model.py + requirements.txt to GitHub, then
               share.streamlit.io -> New app -> main file = app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from credcheck_model import (build_default_model, prepare, GROUPS,
                             FIRM_TYPES, INDUSTRIES, REGIONS)

st.set_page_config(page_title="CredCheck — Adaptive Trade-Credit Limits", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Training the model (first load only)…")
def get_model():
    model, _ = build_default_model()
    return model


model = get_model()
TOGGLE_GROUPS = ["trade", "gst", "bank", "bureau", "fin"]   # firmographics always present
GROUP_LABEL = {"trade": "Trade history with anchor", "gst": "GST / GSTN",
               "bank": "Bank statements (AA)", "bureau": "Credit bureau",
               "fin": "Financial statements"}


def risk_grade(pd_val):
    return ("A (low)" if pd_val < 0.05 else "B (moderate)" if pd_val < 0.12
            else "C (elevated)" if pd_val < 0.25 else "D (high)")


def reason_codes(row, scored):
    notes = []
    active = [g for g in GROUPS if g != "all"
              and (1 - row[GROUPS[g]].isna().mean(axis=1).iloc[0]) >= 0.30]
    gated = [GROUP_LABEL[g] for g in TOGGLE_GROUPS if g not in active]
    notes.append(f"✅ Scored using {len([g for g in active if g!='firmo'])} evidence source(s): "
                 + ", ".join(GROUP_LABEL.get(g, g) for g in active if g != "firmo"))
    if gated:
        notes.append("⚪ No data (expert switched off, **not** penalised): " + ", ".join(gated))
    conf = scored["confidence"].iloc[0]
    if conf < 0.70:
        notes.append("🔎 Thin data → limit sized **conservatively** (lower confidence, not higher risk).")
    r = row.iloc[0]
    if pd.notna(r.get("bureau_score")):
        notes.append(("➕ Strong bureau score." if r["bureau_score"] >= 750
                      else "➖ Weak bureau score." if r["bureau_score"] < 650 else "• Average bureau score."))
    if pd.notna(r.get("trade_ontime_ratio")):
        if r["trade_ontime_ratio"] < 0.70:
            notes.append("➖ Frequent late payments to the anchor.")
        elif r["trade_ontime_ratio"] >= 0.90:
            notes.append("➕ Excellent on-time payment record.")
    if pd.notna(r.get("fin_debt_to_equity")) and r["fin_debt_to_equity"] > 3:
        notes.append("➖ High leverage (debt-to-equity > 3).")
    if pd.notna(r.get("bank_bounce_rate")) and r["bank_bounce_rate"] > 0.10:
        notes.append("➖ Elevated cheque/auto-debit bounce rate.")
    return notes


st.title("CredCheck — Adaptive Trade-Credit Limit Engine")
st.caption("Anchor-led supply-chain credit. The risk estimate does **not** punish missing data — "
           "absent evidence lowers *confidence* and sizes the limit conservatively instead.")

tab_single, tab_batch, tab_about = st.tabs(["Score one firm", "Score a CSV", "How it works"])

# ----------------------------------------------------------------------------
# SINGLE-FIRM TAB
# ----------------------------------------------------------------------------
with tab_single:
    st.markdown("#### Firm profile")
    c1, c2, c3, c4 = st.columns(4)
    firm_type = c1.selectbox("Legal structure", FIRM_TYPES, index=0)
    industry = c2.selectbox("Industry", INDUSTRIES, index=0)
    region = c3.selectbox("Region", REGIONS, index=0)
    vintage = c4.number_input("Years in business", 0.2, 30.0, 5.0, 0.5)

    row = {"firm_type": firm_type, "industry": industry, "region": region,
           "firmo_vintage_years": vintage}
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
            d, e, f = st.columns(3)
            row["trade_payment_cv"] = d.slider("Payment-amount volatility (CV)", 0.0, 1.0, 0.20)
            row["trade_tenure_months"] = e.number_input("Relationship tenure (months)", 0, 360, 24)
            row["trade_dispute_rate"] = f.slider("Dispute/return rate", 0.0, 0.6, 0.03)
        if have["gst"]:
            st.markdown("**GST**")
            a, b, c, d = st.columns(4)
            row["gst_turnover_monthly"] = a.number_input("Monthly GST turnover (₹)", 0, 100_000_000, 500_000, step=10_000)
            row["gst_turnover_trend"] = b.slider("Turnover trend (YoY)", -0.3, 0.5, 0.08)
            row["gst_filing_regularity"] = c.slider("Filing regularity", 0.0, 1.0, 0.92)
            row["gst_counterparty_conc"] = d.slider("Counterparty concentration", 0.0, 1.0, 0.45)
        if have["bank"]:
            st.markdown("**Bank statements**")
            a, b, c, d = st.columns(4)
            row["bank_avg_balance"] = a.number_input("Average balance (₹)", 0, 50_000_000, 150_000, step=10_000)
            row["bank_inflow_outflow_ratio"] = b.slider("Inflow/outflow ratio", 0.6, 2.0, 1.05)
            row["bank_balance_cv"] = c.slider("Balance volatility (CV)", 0.0, 1.5, 0.30)
            row["bank_bounce_rate"] = d.slider("Bounce rate", 0.0, 0.5, 0.02)
        if have["bureau"]:
            st.markdown("**Credit bureau**")
            a, b, c, d = st.columns(4)
            row["bureau_score"] = a.slider("Bureau score", 300, 900, 740)
            row["bureau_num_loans"] = b.number_input("Existing loans", 0, 20, 2)
            row["bureau_dpd_max_12m"] = c.number_input("Max DPD (12m)", 0, 180, 0)
            row["bureau_enquiry_velocity"] = d.number_input("Enquiry velocity", 0, 25, 2)
        if have["fin"]:
            st.markdown("**Financial statements**")
            a, b, c = st.columns(3)
            row["fin_current_ratio"] = a.slider("Current ratio", 0.2, 5.0, 1.6)
            row["fin_debt_to_equity"] = b.slider("Debt-to-equity", 0.05, 8.0, 1.0)
            row["fin_net_margin"] = c.slider("Net margin", -0.2, 0.4, 0.08)
            d, e = st.columns(2)
            row["fin_interest_coverage"] = d.slider("Interest coverage", 0.1, 25.0, 4.0)
            row["fin_net_worth"] = e.number_input("Net worth (₹)", 0, 500_000_000, 2_000_000, step=100_000)

    # any toggled-off group stays NaN (already initialised above)
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

# ----------------------------------------------------------------------------
# BATCH TAB
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown("Upload a CSV of firms (any of the standard columns; missing ones are fine). "
                "Try the sample files like `thinfile_proprietorships.csv`.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        out = model.score(prepare(raw))
        result = pd.concat([raw.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        st.dataframe(result, use_container_width=True, height=380)
        a, b, c = st.columns(3)
        a.metric("Firms scored", len(out))
        b.metric("Median limit", f"₹{out['recommended_limit'].median():,.0f}")
        c.metric("Median confidence", f"{out['confidence'].median()*100:.0f}%")
        st.download_button("Download scored CSV", result.to_csv(index=False),
                           "scored.csv", "text/csv")

# ----------------------------------------------------------------------------
# ABOUT TAB
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**The principle.** Missing data is not bad data. A firm we know *less* about is scored with
*more uncertainty* → a more conservative limit, never a worse risk estimate.

**How.** One gradient-boosting *expert* per evidence source (trade, GST, bank, bureau,
financials) plus a generalist. For each firm, only the experts whose data is present are
activated; their scores are blended by `coverage × reliability`, calibrated to a real
probability, and paired with a *confidence* that falls as data thins. The limit engine then
sizes a transparent line: `capacity × risk_multiplier × confidence`, capped by affordability
and concentration limits.

**Try it:** in *Score one firm*, switch off *Financial statements* and *Credit bureau* —
the probability of default barely moves, but confidence drops and the limit becomes more
cautious. That is the adaptive behaviour the model is built for.
""")
