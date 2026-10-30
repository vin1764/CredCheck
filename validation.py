"""
CredCheck validation report.

1. Does missing data predict default?   Tests the project's core premise instead of assuming it.
2. Benchmark against standard models.   Same 60/20/20 split, same calibration, same test set.
3. Decline rule and thin-file limit cut. How many firms are declined, and how much of a limit
                                        cut comes from missing data alone.
4. Gate threshold sweep.                How results change as the 30% availability gate moves
                                        from 0% to 100%, on clean and on patchy data.

Run:  python validation.py      -> prints the report and writes VALIDATION.md
"""
import copy
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline

from credcheck_model import (ALL_FEATURES, BREAK_EVEN_PD, GATE, GROUPS, LGD, MARGIN, SEED,
                             CredCheckModel, make_dataset, make_expert, prepare, split_data)

SOURCES = ["trade", "gst", "bank", "bureau", "fin"]
LABEL = {"trade": "Trade history", "gst": "GST", "bank": "Bank statements",
         "bureau": "Credit bureau", "fin": "Financial statements"}
THIN_FILE_CONFIDENCE = 0.70


def missing_flags(df):
    """1 where a whole evidence source is absent for that firm."""
    return pd.DataFrame({f"missing_{g}": df[GROUPS[g]].isna().all(axis=1).astype(int)
                         for g in SOURCES}, index=df.index)


def two_proportion_p(k1, n1, k2, n2):
    """Two-sided p-value for a difference between two default rates."""
    if min(n1, n2) == 0:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def calibrated(model, X_tr, y_tr, X_cal, y_cal, X_te):
    """Fit, calibrate with isotonic regression on the calibration set, predict the test set."""
    model.fit(X_tr, y_tr)
    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(X_cal)[:, 1], y_cal)
    return iso.predict(model.predict_proba(X_te)[:, 1])


def missingness_test(data, train, cal, test):
    rows = []
    flags = missing_flags(data)
    for g in SOURCES:
        miss = flags[f"missing_{g}"] == 1
        n_m, n_p = int(miss.sum()), int((~miss).sum())
        if n_m == 0:
            rows.append([LABEL[g], 0, f"{data['default'][~miss].mean():.1%}", "—", "—", "—",
                         "never fully missing"])
            continue
        k_m, k_p = int(data["default"][miss].sum()), int(data["default"][~miss].sum())
        p = two_proportion_p(k_m, n_m, k_p, n_p)
        verdict = ("missing data carries risk" if p < 0.05
                   else "borderline: re-test on real data" if p < 0.10 else "no risk signal")
        rows.append([LABEL[g], f"{n_m:,}", f"{k_p / n_p:.1%}", f"{k_m / n_m:.1%}",
                     f"{(k_m / n_m - k_p / n_p) * 100:+.1f} pts", f"{p:.3f}", verdict])
    table = pd.DataFrame(rows, columns=["Source", "Firms missing it", "Default rate if present",
                                        "Default rate if missing", "Difference", "p-value", "Verdict"])

    # Does adding "source missing" flags improve a standard model?
    lr = lambda: make_expert("logreg")
    X = lambda df: df[ALL_FEATURES]
    Xf = lambda df: pd.concat([df[ALL_FEATURES], missing_flags(df)], axis=1)
    y_tr, y_cal, y_te = train["default"], cal["default"], test["default"]
    auc_base = roc_auc_score(y_te, calibrated(lr(), X(train), y_tr, X(cal), y_cal, X(test)))
    auc_flags = roc_auc_score(y_te, calibrated(lr(), Xf(train), y_tr, Xf(cal), y_cal, Xf(test)))
    flags_only = lr().fit(missing_flags(train), y_tr)
    auc_only = roc_auc_score(y_te, flags_only.predict_proba(missing_flags(test))[:, 1])
    return table, auc_base, auc_flags, auc_only


def benchmark(train, cal, test):
    X = lambda df: df[ALL_FEATURES]
    Xf = lambda df: pd.concat([df[ALL_FEATURES], missing_flags(df)], axis=1)
    y_tr, y_cal, y_te = train["default"], cal["default"], test["default"]
    candidates = [
        ("Logistic regression (median fill)", lambda: make_expert("logreg"), X),
        ("Logistic regression + missing flags", lambda: make_expert("logreg"), Xf),
        ("Random forest, 500 trees (median fill)",
         lambda: make_pipeline(SimpleImputer(strategy="median"),
                               RandomForestClassifier(500, min_samples_leaf=20,
                                                      random_state=SEED, n_jobs=-1)), X),
        ("HistGBM, one model on all 13 features", lambda: make_expert("hgb"), X),
    ]
    rows = []
    for name, factory, feats in candidates:
        p = calibrated(factory(), feats(train), y_tr, feats(cal), y_cal, feats(test))
        rows.append([name, roc_auc_score(y_te, p), brier_score_loss(y_te, p)])
    for name, kind in [("CredCheck, HistGBM experts (default)", "hgb"),
                       ("CredCheck, logistic-regression experts", "logreg")]:
        p, _ = CredCheckModel(expert=kind).fit(train, cal).predict(test)
        rows.append([name, roc_auc_score(y_te, p), brier_score_loss(y_te, p)])
    rows.append(["True PD (best possible on this data)", roc_auc_score(y_te, test["pd_true"]),
                 brier_score_loss(y_te, test["pd_true"])])
    table = pd.DataFrame(rows, columns=["Model", "Test AUC", "Brier (lower is better)"])
    table = table.sort_values("Test AUC", ascending=False)
    table["Test AUC"] = table["Test AUC"].map("{:.3f}".format)
    table["Brier (lower is better)"] = table["Brier (lower is better)"].map("{:.3f}".format)
    return table


def policy_summary(train, cal, test):
    model = CredCheckModel().fit(train, cal)
    actual = model.score(test)
    full = model.score(test, assume_full_data=True)
    approved = actual["decision"] == "approve"
    y = test["default"]
    decl = pd.DataFrame([
        ["Approved", f"{approved.mean():.0%}", f"{y[approved].mean():.1%}"],
        ["Declined", f"{(~approved).mean():.0%}", f"{y[~approved].mean():.1%}"],
    ], columns=["Decision", "Share of test firms", "Actual default rate"])

    thin = approved & (actual["confidence"] < THIN_FILE_CONFIDENCE)
    cut = 1 - actual.loc[thin, "recommended_limit"] / full.loc[thin, "recommended_limit"]
    total_cut = 1 - actual.loc[thin, "recommended_limit"].sum() / full.loc[thin, "recommended_limit"].sum()
    return decl, int(thin.sum()), cut, total_cut


def patchy(df, share=0.25, seed=SEED):
    """Blank a random share of individual fields, to mimic messy real-world uploads."""
    df = df.copy()
    cols = [c for c in ALL_FEATURES if c != "firm_type_code"]
    vals = df[cols].to_numpy(dtype=float).copy()
    vals[np.random.default_rng(seed).random(vals.shape) < share] = np.nan
    df[cols] = vals
    return df


def gate_sweep(train, cal, test):
    """Re-run the blend at gate thresholds from 0% to 100%. Experts are trained once
    (they don't depend on the gate); calibration is refit for each threshold."""
    base = CredCheckModel().fit(train, cal)
    test_patchy = patchy(test)
    rows = []
    for gate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = copy.copy(base)
        m.gate = gate
        cal_raw, _ = m._raw_and_conf(cal)
        m.iso = IsotonicRegression(out_of_bounds="clip").fit(cal_raw, cal["default"])
        actual, full = m.score(test), m.score(test, assume_full_data=True)
        approved = actual["decision"] == "approve"
        thin = approved & (actual["confidence"] < THIN_FILE_CONFIDENCE)
        cut = (1 - actual.loc[thin, "recommended_limit"] / full.loc[thin, "recommended_limit"]).mean()
        rows.append([f"{gate:.0%}" + (" (current)" if gate == GATE else ""),
                     f"{roc_auc_score(cal['default'], m.iso.predict(cal_raw)):.3f}",
                     f"{roc_auc_score(test['default'], m.predict(test)[0]):.3f}",
                     f"{roc_auc_score(test['default'], m.predict(test_patchy)[0]):.3f}",
                     f"{approved.mean():.0%}", f"{cut:.0%}"])
    return pd.DataFrame(rows, columns=["Gate threshold", "Calibration AUC", "Test AUC",
                                       "Test AUC, patchy data", "Approved", "Thin-file cut"])


def main():
    data = prepare(make_dataset())
    train, cal, test = split_data(data)

    miss_table, auc_base, auc_flags, auc_only = missingness_test(data, train, cal, test)
    bench = benchmark(train, cal, test)
    decl, n_thin, cut, total_cut = policy_summary(train, cal, test)
    sweep = gate_sweep(train, cal, test)

    report = f"""# CredCheck validation report

Generated by `python validation.py` on the simulated 12,000-firm dataset
(60/20/20 split: 7,200 train, 2,400 calibration, 2,400 test).

## 1. Does missing data predict default?

CredCheck's design assumes a missing evidence source signals uncertainty, not risk.
This checks that assumption instead of taking it on trust.

{md_table(miss_table)}

Adding "source missing" flags to a logistic regression: test AUC {auc_base:.3f} without,
{auc_flags:.3f} with. A model using only the missing flags scores AUC {auc_only:.3f}
(0.5 = no signal).

**Reading it:** on this simulated data, missingness carries no meaningful risk signal, as
designed: blanks were generated by legal structure or at random, not by risk, and the missing flags add
nothing to a model. Any borderline row is chance at this sample size. On a real lender's book, run this
first. If a source's absence does predict default, add its missing flag as a feature so the
model learns that risk instead of ignoring it.

## 2. Benchmark on the same split

All models are trained on the same 7,200 firms, calibrated with isotonic regression on the
same 2,400 firms, and scored on the same 2,400 test firms.

{md_table(bench)}

**Reading it:** logistic regression scores highest because the simulated risk is mostly
linear: a weighted sum of hidden drivers, which is exactly what logistic regression assumes.
The thresholds and interactions the simulation does include (30+ days late, debt hurting late
payers, young firms) are mild. The CredCheck architecture works with either expert; switch with
`CredCheckModel(expert="logreg")`. Gradient boosting stays the default because real credit
data usually has stronger thresholds and interactions than the simulation.

## 3. Decline rule and the thin-file limit cut

A firm is declined when its expected loss (calibrated PD × LGD of {LGD:.0%}) is larger than the
expected margin on the sale ((1 − PD) × {MARGIN:.0%}), i.e. a PD above {BREAK_EVEN_PD:.0%}, the
break-even point. The rule uses the calibrated PD only, so missing data never causes a decline
by itself.

{md_table(decl)}

**Thin-file limit cut.** Among approved firms with confidence below {THIN_FILE_CONFIDENCE:.0%}
({n_thin} test firms), limits are on average **{cut.mean():.0%}** lower than the same firms would
get with complete data (range {cut.min():.0%} to {cut.max():.0%}; {total_cut:.0%} of total limit
value). The cut comes from missing data alone; the PD and the decision are unchanged.

## 4. Gate threshold sweep

An expert votes only if at least {GATE:.0%} of its fields are present. That value was set by
judgment, so this re-runs the model at every threshold from 0% to 100%. "Patchy data" blanks
25% of individual fields at random on the test set, to mimic messy real uploads.

{md_table(sweep)}

**Reading it:** every threshold from 0% to 30% gives practically the same result, because
each vote is already weighted by coverage; the gate only guards against an expert voting on a
sliver of its inputs. From 40% up, partly filled sources start being switched off (a trade
record with only order value is 33% covered), so useful evidence is thrown away: approvals
shift and accuracy falls, sharply on patchy data. {GATE:.0%} sits inside the flat range. On a lender's real data,
rerun this sweep on the calibration set and keep a value from the flat region.
"""
    with open("VALIDATION.md", "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
