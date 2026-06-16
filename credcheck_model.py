"""
CredCheck — adaptive trade-credit limit model (reusable module).

Usage as a library:
    from credcheck_model import build_default_model, prepare
    model, data = build_default_model()
    scored = model.score(prepare(my_dataframe))

Usage from the command line (score any CSV with the standard schema):
    python credcheck_model.py path/to/incomplete_firms.csv
    -> writes path/to/incomplete_firms_scored.csv  (pd, confidence, limit per firm)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 7
FIRM_TYPES = ["proprietorship", "partnership", "small_pvt", "large_pvt"]
INDUSTRIES = ["agri_inputs", "dairy", "fmcg_distribution", "general_trade"]
REGIONS = ["north", "south", "east", "west"]
TYPE_MIX = [0.40, 0.25, 0.25, 0.10]
MISSING_P = {                       # P(an entire group is unobserved) by firm type
    "proprietorship": (0.90, 0.35, 0.25),   # (fin, bureau, gst)
    "partnership":    (0.50, 0.15, 0.10),
    "small_pvt":      (0.20, 0.06, 0.04),
    "large_pvt":      (0.03, 0.02, 0.01),
}

# ---- Feature groups -> one expert each.  "all" is a NaN-native generalist. ----
GROUPS = {
    "trade":  ["trade_ontime_ratio", "trade_days_to_pay_vs_terms", "trade_order_vol_monthly",
               "trade_payment_cv", "trade_tenure_months", "trade_dispute_rate"],
    "gst":    ["gst_turnover_monthly", "gst_turnover_trend", "gst_filing_regularity", "gst_counterparty_conc"],
    "bank":   ["bank_avg_balance", "bank_inflow_outflow_ratio", "bank_balance_cv", "bank_bounce_rate"],
    "bureau": ["bureau_score", "bureau_num_loans", "bureau_dpd_max_12m", "bureau_enquiry_velocity"],
    "fin":    ["fin_current_ratio", "fin_debt_to_equity", "fin_net_margin",
               "fin_interest_coverage", "fin_net_worth"],
    "firmo":  ["firmo_vintage_years", "firm_type_code", "industry_code", "region_code"],
}
CAT_COLS = {"firm_type": ("firm_type_code", FIRM_TYPES),
            "industry": ("industry_code", INDUSTRIES),
            "region": ("region_code", REGIONS)}
ALL_FEATURES = sorted({c for cols in GROUPS.values() for c in cols})
GROUPS["all"] = ALL_FEATURES

# Limit-engine policy knobs
CREDIT_PERIOD_MONTHS = 1.0
BUFFER = 1.10
CONCENTRATION_CAP = 8_000_000
FLOOR = 25_000


# ---------------------------------------------------------------------------
# 1. Synthetic data generator
# ---------------------------------------------------------------------------
def make_dataset(n=12000, seed=SEED):
    rng = np.random.default_rng(seed)
    ftype = rng.choice(FIRM_TYPES, size=n, p=TYPE_MIX)
    industry = rng.choice(INDUSTRIES, size=n)
    region = rng.choice(REGIONS, size=n)
    vintage = np.clip(rng.gamma(2.2, 2.2, n), 0.2, 30).round(1)

    z_pay = rng.normal(0, 1, n); z_cash = rng.normal(0, 1, n); z_lev = rng.normal(0, 1, n)
    z_bur = rng.normal(0, 1, n); z_grw = rng.normal(0, 1, n)
    latent = (-0.95 * z_pay - 0.70 * z_cash + 1.05 * z_lev
              - 0.85 * z_bur - 0.45 * z_grw + 0.30 * rng.normal(0, 1, n))
    pd_true = 1 / (1 + np.exp(-(latent - 2.3)))
    default = (rng.random(n) < pd_true).astype(int)

    df = pd.DataFrame({"firm_type": ftype, "industry": industry, "region": region})
    df["trade_ontime_ratio"] = np.clip(0.78 + 0.12 * z_pay + 0.05 * rng.normal(0, 1, n), 0, 1).round(3)
    df["trade_days_to_pay_vs_terms"] = (3 - 6 * z_pay + 4 * rng.normal(0, 1, n)).round(1)
    base_vol = np.exp(rng.normal(12.6, 1.0, n))
    df["trade_order_vol_monthly"] = base_vol.round(0)
    df["trade_payment_cv"] = np.clip(0.25 - 0.06 * z_cash + 0.05 * rng.normal(0, 1, n), 0.02, 1).round(3)
    df["trade_tenure_months"] = np.clip(vintage * 12 * rng.uniform(0.3, 1.0, n), 1, 360).round(0)
    df["trade_dispute_rate"] = np.clip(0.04 - 0.015 * z_pay + 0.02 * rng.normal(0, 1, n), 0, 0.6).round(3)

    df["gst_turnover_monthly"] = (base_vol * rng.uniform(1.2, 2.5, n)).round(0)
    df["gst_turnover_trend"] = (0.06 + 0.10 * z_grw + 0.04 * rng.normal(0, 1, n)).round(3)
    df["gst_filing_regularity"] = np.clip(0.85 + 0.10 * z_pay + 0.05 * rng.normal(0, 1, n), 0, 1).round(3)
    df["gst_counterparty_conc"] = np.clip(0.45 - 0.05 * z_cash + 0.08 * rng.normal(0, 1, n), 0.05, 0.95).round(3)

    df["bank_avg_balance"] = np.clip(base_vol * rng.uniform(0.1, 0.6, n) * (1 + 0.3 * z_cash), 2000, None).round(0)
    df["bank_inflow_outflow_ratio"] = np.clip(1.02 + 0.10 * z_cash + 0.05 * rng.normal(0, 1, n), 0.6, 2).round(3)
    df["bank_balance_cv"] = np.clip(0.30 - 0.08 * z_cash + 0.05 * rng.normal(0, 1, n), 0.02, 1.5).round(3)
    df["bank_bounce_rate"] = np.clip(0.03 - 0.012 * z_pay + 0.015 * rng.normal(0, 1, n), 0, 0.5).round(3)

    df["bureau_score"] = np.clip(720 + 70 * z_bur + 25 * rng.normal(0, 1, n), 300, 900).round(0)
    df["bureau_num_loans"] = np.clip(rng.poisson(2.2, n), 0, 20)
    df["bureau_dpd_max_12m"] = np.clip((30 * np.maximum(0, -z_bur) + 10 * rng.normal(0, 1, n)).round(0), 0, 180)
    df["bureau_enquiry_velocity"] = np.clip(rng.poisson(np.maximum(0.5, 2 - z_bur), n), 0, 25)

    df["fin_current_ratio"] = np.clip(1.5 - 0.42 * z_lev + 0.10 * rng.normal(0, 1, n), 0.2, 5).round(3)
    df["fin_debt_to_equity"] = np.clip(1.0 + 0.75 * z_lev + 0.12 * rng.normal(0, 1, n), 0.05, 8).round(3)
    df["fin_net_margin"] = np.clip(0.07 - 0.045 * z_lev + 0.012 * rng.normal(0, 1, n), -0.2, 0.4).round(3)
    df["fin_interest_coverage"] = np.clip(4 - 2.2 * z_lev + 0.5 * rng.normal(0, 1, n), 0.1, 25).round(2)
    df["fin_net_worth"] = np.clip(base_vol * rng.uniform(2, 8, n) * (1 - 0.25 * z_lev), 10000, None).round(0)
    df["firmo_vintage_years"] = vintage

    grp_fin, grp_bur, grp_gst = GROUPS["fin"], GROUPS["bureau"], GROUPS["gst"]
    grp_trade = GROUPS["trade"]
    for t in FIRM_TYPES:
        m = df["firm_type"].values == t
        p_fin, p_bur, p_gst = MISSING_P[t]
        for grp, p in [(grp_fin, p_fin), (grp_bur, p_bur), (grp_gst, p_gst)]:
            df.loc[m & (rng.random(n) < p), grp] = np.nan
    df.loc[rng.random(n) < 0.09, grp_trade] = np.nan       # cold-start: no trade history

    df["default"] = default
    df["pd_true"] = pd_true.round(4)
    return df


# ---------------------------------------------------------------------------
# 2. Prepare any dataframe for scoring (encode categoricals, ensure columns)
# ---------------------------------------------------------------------------
def prepare(df):
    df = df.copy()
    for raw, (code, cats) in CAT_COLS.items():
        if raw in df.columns:
            c = pd.Categorical(df[raw], categories=cats).codes.astype(float)
            c[c < 0] = np.nan                      # unknown / missing -> NaN
            df[code] = c
        elif code not in df.columns:
            df[code] = np.nan
    for col in ALL_FEATURES:                       # ensure every model feature exists
        if col not in df.columns:
            df[col] = np.nan
    return df


# ---------------------------------------------------------------------------
# 3. The model
# ---------------------------------------------------------------------------
def _coverage(df, cols):
    return 1.0 - df[cols].isna().mean(axis=1).values


class CredCheckModel:
    def __init__(self):
        self.experts, self.reliability, self.iso, self.prior_pd, self.max_w = {}, {}, None, 0.15, 1.0

    def fit(self, train, cal):
        self.prior_pd = float(train["default"].mean())
        for g, cols in GROUPS.items():
            clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400,
                                                 l2_regularization=1.0, random_state=SEED)
            clf.fit(train[cols], train["default"])
            self.experts[g] = clf
            auc = roc_auc_score(cal["default"], clf.predict_proba(cal[cols])[:, 1])
            self.reliability[g] = max(0.0, (auc - 0.5) * 2)
        self.reliability["all"] *= 0.6
        self.max_w = sum(self.reliability.values())
        cal_raw, _ = self._raw_and_conf(cal)
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(cal_raw, cal["default"])
        return self

    def _raw_and_conf(self, df):
        n = len(df); num = np.zeros(n); den = np.zeros(n); cov_w = np.zeros(n)
        for g, cols in GROUPS.items():
            cov = _coverage(df, cols)
            w = np.where(cov >= 0.30, cov * self.reliability[g], 0.0)   # availability gate
            pd_g = self.experts[g].predict_proba(df[cols])[:, 1]
            num += w * pd_g; den += w; cov_w += w
        raw = np.where(den > 0, num / np.maximum(den, 1e-9), self.prior_pd)
        confidence = 0.40 + 0.60 * np.clip(cov_w / max(self.max_w, 1e-9), 0.0, 1.0)
        return raw, confidence

    def predict(self, df):
        raw, conf = self._raw_and_conf(df)
        return self.iso.predict(raw), conf

    def score(self, df):
        """Return PD, confidence, conservative PD, and recommended limit per firm."""
        pd_hat, conf = self.predict(df)
        run_rate = df["trade_order_vol_monthly"].copy()
        run_rate = run_rate.fillna(df["gst_turnover_monthly"] * 0.5).fillna(50_000).to_numpy(dtype=float)
        capacity = run_rate * CREDIT_PERIOD_MONTHS * BUFFER
        pd_cons = np.clip(pd_hat + (1 - conf) * pd_hat * 0.8, 0, 0.95)
        risk_mult = np.clip(1 - pd_cons, 0.05, 1.0)
        bal = df["bank_avg_balance"].to_numpy(dtype=float)
        bal = np.where(np.isnan(bal), run_rate * 0.15, bal)
        afford = np.maximum(bal * 6.0, run_rate * 0.5)
        limit = np.clip(capacity * risk_mult * (0.7 + 0.3 * conf), FLOOR,
                        np.minimum(afford, CONCENTRATION_CAP))
        out = pd.DataFrame({
            "pd": pd_hat.round(4), "confidence": conf.round(3),
            "pd_conservative": pd_cons.round(4), "recommended_limit": np.round(limit, -2),
        }, index=df.index)
        return out


def build_default_model(n=12000, seed=SEED):
    """Train the model on a fresh synthetic dataset and return (model, data)."""
    data = prepare(make_dataset(n=n, seed=seed))
    train, tmp = train_test_split(data, test_size=0.40, random_state=seed, stratify=data["default"])
    cal, _ = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=tmp["default"])
    model = CredCheckModel().fit(train, cal)
    return model, data


# ---------------------------------------------------------------------------
# 4. CLI: score an incomplete CSV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    path = sys.argv[1]
    raw_df = pd.read_csv(path)
    model, _ = build_default_model()
    scored = model.score(prepare(raw_df))
    result = pd.concat([raw_df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    out_path = path.rsplit(".", 1)[0] + "_scored.csv"
    result.to_csv(out_path, index=False)
    n_active = (scored["confidence"] > 0.4).sum()
    print(f"Scored {len(scored)} firms -> {out_path}")
    print(f"  median PD={scored['pd'].median():.3f}  median confidence={scored['confidence'].median():.3f}"
          f"  median limit=Rs{scored['recommended_limit'].median():,.0f}")
    if "default" in raw_df.columns and raw_df["default"].nunique() > 1:
        print(f"  AUC on this file = {roc_auc_score(raw_df['default'], scored['pd']):.3f}")
