"""
CredCheck — adaptive trade-credit limit model (reusable module, lean feature set).

13 features across 6 evidence groups (2-3 each), one expert per group + a generalist.

Library use:
    from credcheck_model import build_default_model, prepare
    model, data = build_default_model()
    scored = model.score(prepare(my_dataframe))

CLI (score any incomplete CSV with the standard schema):
    python credcheck_model.py path/to/firms.csv   ->   path/to/firms_scored.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 7
FIRM_TYPES = ["proprietorship", "partnership", "small_pvt", "large_pvt"]
TYPE_MIX = [0.40, 0.25, 0.25, 0.10]
MISSING_P = {                       # P(an entire group is unobserved) by firm type
    "proprietorship": (0.90, 0.35, 0.25),   # (fin, bureau, gst)
    "partnership":    (0.50, 0.15, 0.10),
    "small_pvt":      (0.20, 0.06, 0.04),
    "large_pvt":      (0.03, 0.02, 0.01),
}

# ---- Six evidence groups -> one expert each. "all" is a NaN-native generalist. ----
GROUPS = {
    "trade":  ["trade_ontime_ratio", "trade_days_to_pay_vs_terms", "trade_order_vol_monthly"],
    "gst":    ["gst_turnover_monthly", "gst_turnover_trend"],
    "bank":   ["bank_avg_balance", "bank_bounce_rate"],
    "bureau": ["bureau_score", "bureau_dpd_max_12m"],
    "fin":    ["fin_debt_to_equity", "fin_current_ratio"],
    "firmo":  ["firm_type_code", "firmo_vintage_years"],
}
CAT_COLS = {"firm_type": ("firm_type_code", FIRM_TYPES)}
ALL_FEATURES = sorted({c for cols in GROUPS.values() for c in cols})
GROUPS["all"] = ALL_FEATURES

# Limit-engine policy knobs
CREDIT_PERIOD_MONTHS = 1.0
BUFFER = 1.10
CONCENTRATION_CAP = 8_000_000
FLOOR = 25_000
GST_PURCHASE_FRACTION = 0.40        # GST turnover -> implied purchasing scale for sizing

# Decline policy (break-even): approve only if the expected margin on the sale covers the
# expected loss, i.e. (1 - PD) x MARGIN >= PD x LGD, which means PD <= MARGIN / (MARGIN + LGD).
# It uses the calibrated PD, never the confidence-inflated one, so missing data can
# shrink a limit but can never by itself cause a decline.
LGD = 0.60                          # loss given default, unsecured MSME credit (policy assumption)
MARGIN = 0.20                       # supplier's margin on a credit sale; set per supplier
BREAK_EVEN_PD = MARGIN / (MARGIN + LGD)   # 25% at these values: decline above it

# Availability gate: an expert votes only if at least this share of its fields is present.
# Votes are already weighted by coverage, so this only guards against votes on a sliver of
# inputs. validation.py sweeps 0-100%: results are identical from 0% to 30%.
GATE = 0.30


# ---------------------------------------------------------------------------
# 1. Synthetic data generator
# ---------------------------------------------------------------------------
def make_dataset(n=12000, seed=SEED):
    rng = np.random.default_rng(seed)
    ftype = rng.choice(FIRM_TYPES, size=n, p=TYPE_MIX)
    vintage = np.clip(rng.gamma(2.2, 2.2, n), 0.2, 30).round(1)

    # True hidden risk drivers (higher = safer, except leverage where higher = worse)
    z_pay = rng.normal(0, 1, n); z_cash = rng.normal(0, 1, n); z_lev = rng.normal(0, 1, n)
    z_bur = rng.normal(0, 1, n); z_grw = rng.normal(0, 1, n)
    latent = (-0.90 * z_pay - 0.80 * z_cash + 0.90 * z_lev
              - 0.90 * z_bur - 0.60 * z_grw + 0.30 * rng.normal(0, 1, n))
    pd_true = 1 / (1 + np.exp(-(latent - 2.3)))
    default = (rng.random(n) < pd_true).astype(int)

    base_vol = np.exp(rng.normal(12.6, 1.0, n))            # monthly order value, INR
    df = pd.DataFrame({"firm_type": ftype})
    # Trade history with the anchor (payment discipline + scale)
    df["trade_ontime_ratio"] = np.clip(0.82 + 0.12 * z_pay + 0.05 * rng.normal(0, 1, n), 0, 1).round(3)
    df["trade_days_to_pay_vs_terms"] = (2 - 7 * z_pay + 4 * rng.normal(0, 1, n)).round(1)
    df["trade_order_vol_monthly"] = base_vol.round(0)
    # GST (scale + growth)
    df["gst_turnover_monthly"] = (base_vol * rng.uniform(1.2, 2.5, n)).round(0)
    df["gst_turnover_trend"] = (0.06 + 0.12 * z_grw + 0.04 * rng.normal(0, 1, n)).round(3)
    # Bank statements (liquidity + red flags)
    df["bank_avg_balance"] = np.clip(base_vol * rng.uniform(0.1, 0.6, n) * (1 + 0.35 * z_cash), 2000, None).round(0)
    df["bank_bounce_rate"] = np.clip(0.035 - 0.02 * z_cash - 0.01 * z_pay + 0.015 * rng.normal(0, 1, n), 0, 0.5).round(3)
    # Credit bureau
    df["bureau_score"] = np.clip(720 + 75 * z_bur + 25 * rng.normal(0, 1, n), 300, 900).round(0)
    df["bureau_dpd_max_12m"] = np.clip((30 * np.maximum(0, -z_bur) + 8 * rng.normal(0, 1, n)).round(0), 0, 180)
    # Financial statements (leverage + liquidity)
    df["fin_debt_to_equity"] = np.clip(1.0 + 0.80 * z_lev + 0.15 * rng.normal(0, 1, n), 0.05, 8).round(3)
    df["fin_current_ratio"] = np.clip(1.6 - 0.45 * z_lev + 0.12 * rng.normal(0, 1, n), 0.2, 5).round(3)
    # Firmographics
    df["firmo_vintage_years"] = vintage

    # Structural missingness by firm type, + cold-start with no trade history
    for t in FIRM_TYPES:
        m = df["firm_type"].values == t
        p_fin, p_bur, p_gst = MISSING_P[t]
        for grp, p in [(GROUPS["fin"], p_fin), (GROUPS["bureau"], p_bur), (GROUPS["gst"], p_gst)]:
            df.loc[m & (rng.random(n) < p), grp] = np.nan
    df.loc[rng.random(n) < 0.09, GROUPS["trade"]] = np.nan

    df["default"] = default
    df["pd_true"] = pd_true.round(4)
    return df


# ---------------------------------------------------------------------------
# 2. Prepare any dataframe for scoring
# ---------------------------------------------------------------------------
def prepare(df):
    df = df.copy()
    code, cats = "firm_type_code", FIRM_TYPES
    if "firm_type" in df.columns:
        c = pd.Categorical(df["firm_type"], categories=cats).codes.astype(float)
        c[c < 0] = np.nan
        df[code] = c
    elif code not in df.columns:
        df[code] = np.nan
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ---------------------------------------------------------------------------
# 3. The model
# ---------------------------------------------------------------------------
def _coverage(df, cols):
    return 1.0 - df[cols].isna().mean(axis=1).values


def make_expert(kind="hgb"):
    """One expert model. 'hgb' = gradient boosting (default); 'logreg' = logistic regression."""
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400,
                                              l2_regularization=1.0, random_state=SEED)
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(f"unknown expert kind: {kind}")


class CredCheckModel:
    def __init__(self, expert="hgb", gate=GATE):
        self.expert_kind = expert
        self.gate = gate
        self.experts, self.reliability, self.iso, self.prior_pd, self.max_w = {}, {}, None, 0.15, 1.0

    def fit(self, train, cal):
        self.prior_pd = float(train["default"].mean())
        for g, cols in GROUPS.items():
            clf = make_expert(self.expert_kind)
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
            w = np.where(cov >= self.gate, cov * self.reliability[g], 0.0)   # availability gate
            pd_g = self.experts[g].predict_proba(df[cols])[:, 1]
            num += w * pd_g; den += w; cov_w += w
        raw = np.where(den > 0, num / np.maximum(den, 1e-9), self.prior_pd)
        confidence = 0.40 + 0.60 * np.clip(cov_w / max(self.max_w, 1e-9), 0.0, 1.0)
        return raw, confidence

    def predict(self, df):
        raw, conf = self._raw_and_conf(df)
        return self.iso.predict(raw), conf

    def score(self, df, assume_full_data=False):
        """Return PD, confidence, conservative PD, expected loss and margin, decision and limit per firm.

        assume_full_data=True scores the same firms as if confidence were 100%,
        which isolates how much of a limit cut comes from missing data alone.
        """
        pd_hat, conf = self.predict(df)
        if assume_full_data:
            conf = np.ones_like(conf)
        # Capacity sized from the stronger of anchor orders or GST-implied purchasing scale.
        trade_rr = df["trade_order_vol_monthly"]
        gst_rr = df["gst_turnover_monthly"] * GST_PURCHASE_FRACTION
        run_rate = pd.concat([trade_rr, gst_rr], axis=1).max(axis=1).fillna(50_000).to_numpy(dtype=float)
        capacity = run_rate * CREDIT_PERIOD_MONTHS * BUFFER
        pd_cons = np.clip(pd_hat + (1 - conf) * pd_hat * 0.8, 0, 0.95)
        risk_mult = np.clip(1 - pd_cons, 0.05, 1.0)
        bal = df["bank_avg_balance"].to_numpy(dtype=float)
        bal = np.where(np.isnan(bal), run_rate * 0.15, bal)
        afford = np.maximum(bal * 6.0, run_rate * 0.5)
        limit = np.clip(capacity * risk_mult * (0.7 + 0.3 * conf), FLOOR,
                        np.minimum(afford, CONCENTRATION_CAP))
        # Decline rule (break-even): expected margin must cover expected loss. Uses the
        # calibrated PD only, so thin data alone never declines a firm.
        el_rate = pd_hat * LGD
        margin_rate = (1 - pd_hat) * MARGIN
        approve = margin_rate >= el_rate
        limit = np.where(approve, limit, 0.0)
        return pd.DataFrame({
            "pd": pd_hat.round(4), "confidence": conf.round(3),
            "pd_conservative": pd_cons.round(4), "expected_loss_rate": el_rate.round(4),
            "expected_margin_rate": margin_rate.round(4),
            "decision": np.where(approve, "approve", "decline"),
            "recommended_limit": np.round(limit, -2),
        }, index=df.index)


def split_data(data, seed=SEED):
    """60/20/20 stratified split into train, calibration and test sets."""
    train, tmp = train_test_split(data, test_size=0.40, random_state=seed, stratify=data["default"])
    cal, test = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=tmp["default"])
    return train, cal, test


def build_default_model(n=12000, seed=SEED, expert="hgb"):
    data = prepare(make_dataset(n=n, seed=seed))
    train, cal, _ = split_data(data, seed)
    return CredCheckModel(expert=expert).fit(train, cal), data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    path = sys.argv[1]
    raw_df = pd.read_csv(path)
    model, _ = build_default_model()
    scored = model.score(prepare(raw_df))
    result = pd.concat([raw_df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    out_path = path.rsplit(".", 1)[0] + "_scored.csv"
    result.to_csv(out_path, index=False)
    print(f"Scored {len(scored)} firms -> {out_path}")
    approved = scored["decision"] == "approve"
    print(f"  median PD={scored['pd'].median():.3f}  median confidence={scored['confidence'].median():.3f}"
          f"  approved={approved.mean():.0%}"
          f"  median approved limit=Rs{scored.loc[approved, 'recommended_limit'].median():,.0f}")
    if "default" in raw_df.columns and raw_df["default"].nunique() > 1:
        print(f"  AUC on this file = {roc_auc_score(raw_df['default'], scored['pd']):.3f}")
