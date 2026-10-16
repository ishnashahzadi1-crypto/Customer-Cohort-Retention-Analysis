"""
CUSTOMER COHORT RETENTION + 90-DAY REPEAT-PURCHASE PREDICTION  (v2, fixed)
==========================================================================
Run:
    python Retention.py                         (CSV same folder / data folder)
    python Retention.py --data path/to/file.csv
    python Retention.py --trials 10             (quick test run)
    python Retention.py --test-days 90          (size of final Test window)
 
Outputs (in ./outputs/customer_retention):
    figures/01_cohort_and_customer_overview.png   <- PLOT 1
    figures/02_model_evaluation.png               <- PLOT 2
    reports/retention_report.md                   <- written report
    tables/*.csv, models/best_retention_model.joblib
 
What changed vs v1 (all review points fixed):
  1. Split: Dev (purged) + ONE final Test. No separate Validation window, so the
     model is tuned AND refit on all Dev data, and Test sits only 90 days after Dev
     (no 8-month drift gap, Black-Friday-only validation is gone).
  2. `purchase_month` removed (it was only a time proxy with <2 years of data).
  3. CV: 4 purged expanding folds on Dev (bigger first fold), fewer trials (40),
     fold mean +- std reported, so noise between models is visible.
  4. Threshold + calibration are learned from pooled out-of-fold predictions
     (more positives than one validation window). Platt calibration fixes the
     probabilities distorted by class_weight / scale_pos_weight, so Brier is valid.
  5. Test is used ONCE, only for the selected model. Plots/importance for the
     other models use out-of-fold data. Bootstrap 95% CI on Test metrics.
  6. error_score failures are counted and logged (no silent NaN).
  7. Data end is found from order density (a stray order in Sep-2018 no longer
     moves the cutoff); n_categories ignores "unknown"; imputer adds
     missing-indicators; scaler only for Logistic Regression; HistGB early
     stopping disabled (it used a random, non-time validation split).
"""
 
import argparse
import json
import logging
import shutil
import time
import warnings
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from joblib import Memory, Parallel, delayed
from scipy.stats import loguniform, randint, uniform
 
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:  # LightGBM is optional
    HAS_LGBM = False
 
# ----------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------
RANDOM_STATE = 42
FOLLOWUP_DAYS = 90
CV_FOLDS = 4
N_JOBS = -1
TOP_FRAC = 0.10                                              # "contact top 10%" policy
INVALID_STATUSES = {"canceled", "cancelled", "unavailable"}  # single purchase rule
 
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "customer_retention"
 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
sns.set_theme(style="whitegrid", context="notebook")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("retention")
 
NEEDED_COLS = [
    "order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value",
    "order_status", "order_purchase_timestamp", "customer_unique_id", "customer_state",
    "product_category_name_english", "seller_state", "payment_total_value",
    "payment_types", "payment_installments_max", "product_weight_g",
    "product_photos_qty", "product_length_cm", "product_height_cm", "product_width_cm",
]
 
 
def make_dirs(out_dir):
    dirs = {k: out_dir / k for k in ["tables", "figures", "models", "reports"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
 
 
def find_dataset(cli_path):
    if cli_path:
        p = Path(cli_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"File not found: {p}")
    candidates = [
        PROJECT_DIR / "olist_all_files_merged.csv",
        PROJECT_DIR / "Customer.py" / "olist_all_files_merged.csv",
        PROJECT_DIR / "data" / "olist_all_files_merged.csv",
        PROJECT_DIR.parent / "olist_all_files_merged.csv",
        Path("/mnt/user-data/uploads/olist_all_files_merged.csv"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("CSV not found. Use --data path/to/olist_all_files_merged.csv")
 
 
# ----------------------------------------------------------------
# 2. LOAD + ORDER TABLE (one row per order)
# ----------------------------------------------------------------
def load_raw(path):
    df = pd.read_csv(path, usecols=lambda c: c in NEEDED_COLS, low_memory=False)
    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    log.info("Loaded %s rows from %s", f"{len(df):,}", path.name)
    return df
 
 
def build_orders(df):
    df = df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df["order_status"] = df["order_status"].astype("string").str.strip().str.lower()
    df = df.dropna(subset=["order_id", "customer_unique_id", "order_purchase_timestamp"])
    df = df[~df["order_status"].isin(INVALID_STATUSES)]
 
    # merged files can repeat an item row (e.g. one row per payment/review): drop exact repeats
    before = len(df)
    df = df.drop_duplicates(["order_id", "order_item_id"])
    if len(df) != before:
        log.warning("Dropped %d duplicated (order_id, order_item_id) rows", before - len(df))
 
    for c in ["price", "freight_value", "payment_total_value", "payment_installments_max",
              "product_weight_g", "product_photos_qty", "product_length_cm",
              "product_height_cm", "product_width_cm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["price"] < 0, "price"] = np.nan
    df.loc[df["freight_value"] < 0, "freight_value"] = np.nan
    df["volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
 
    for c in ["customer_state", "seller_state", "product_category_name_english", "payment_types"]:
        df[c] = df[c].astype("string").str.strip().fillna("unknown")
    # category count must ignore the "unknown" filler
    df["_cat_known"] = df["product_category_name_english"].mask(
        df["product_category_name_english"] == "unknown")
 
    g = df.groupby("order_id", sort=False)
    multi = int((g["payment_total_value"].nunique() > 1).sum())
    log.info("Orders with >1 distinct payment_total_value: %d (should be ~0)", multi)
 
    orders = g.agg(
        customer_unique_id=("customer_unique_id", "first"),
        purchase_ts=("order_purchase_timestamp", "min"),
        item_count=("order_item_id", "count"),
        price_sum=("price", "sum"),
        freight_sum=("freight_value", "sum"),
        payment_total=("payment_total_value", "first"),
        installments=("payment_installments_max", "first"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("_cat_known", "nunique"),
        weight_sum_g=("product_weight_g", "sum"),
        photos_mean=("product_photos_qty", "mean"),
        volume_mean=("volume_cm3", "mean"),
        category=("product_category_name_english", "first"),
        customer_state=("customer_state", "first"),
        seller_state=("seller_state", "first"),
        payment_type=("payment_types", "first"),
    ).reset_index()
 
    fallback = orders["price_sum"] + orders["freight_sum"]
    orders["order_value"] = orders["payment_total"].where(orders["payment_total"].notna(), fallback)
    orders["freight_ratio"] = orders["freight_sum"] / (orders["price_sum"] + orders["freight_sum"]).replace(0, np.nan)
    orders["same_state"] = (orders["customer_state"] == orders["seller_state"]).astype(int)
    orders["payment_type"] = orders["payment_type"].astype("string").str.split(",").str[0].str.strip()
    orders = orders.sort_values(["purchase_ts", "order_id"]).reset_index(drop=True)
    log.info("Order table: %s orders, %s customers",
             f"{len(orders):,}", f"{orders['customer_unique_id'].nunique():,}")
    return orders
 
 
def find_cutoff(orders):
    """End of observed data, found from order DENSITY (not from the max timestamp).
 
    A few stray orders after the real end of the data would otherwise make recent
    customers look like they have a complete 90-day window when they do not.
    Returns an exclusive end timestamp (midnight after the last 'busy' day).
    """
    daily = orders.groupby(orders["purchase_ts"].dt.normalize())["order_id"].nunique()
    typical = daily.tail(120).median()
    busy = daily[daily >= max(1.0, 0.2 * typical)]
    cutoff = busy.index.max() + pd.Timedelta(days=1)
    stray = int((orders["purchase_ts"] >= cutoff).sum())
    log.info("Data end (by order density): %s | stray orders after it: %d",
             (cutoff - pd.Timedelta(days=1)).date(), stray)
    return cutoff
 
 
def last_full_period(cutoff):
    """Last calendar month that is (almost) complete."""
    last_day = cutoff - pd.Timedelta(days=1)
    p = pd.Period(last_day, "M")
    return p if last_day.day >= last_day.days_in_month - 3 else p - 1
 
 
# ----------------------------------------------------------------
# 3. COHORT ANALYSIS (vectorised)
# ----------------------------------------------------------------
def cohort_analysis(orders, dirs, last_full_p):
    o = orders[["customer_unique_id", "purchase_ts"]].copy()
    o["m"] = o["purchase_ts"].dt.year * 12 + o["purchase_ts"].dt.month
    last_full = last_full_p.year * 12 + last_full_p.month
    o = o[o["m"] <= last_full].copy()          # incomplete final month would fake 0% retention
    o["cohort"] = o.groupby("customer_unique_id")["m"].transform("min")
    o["period"] = o["m"] - o["cohort"]
 
    act = o[["customer_unique_id", "cohort", "period"]].drop_duplicates()
    counts = act.groupby(["cohort", "period"])["customer_unique_id"].nunique().unstack(fill_value=0)
    sizes = counts[0]
    retention = counts.div(sizes, axis=0) * 100
 
    # Cells beyond the end of the data are NOT zero retention: mask them.
    last_m = o["m"].max()
    allowed = (last_m - counts.index.values)[:, None]
    retention = retention.where(retention.columns.values[None, :] <= allowed)
 
    labels = [f"{(c - 1) // 12}-{(c - 1) % 12 + 1:02d}" for c in counts.index]
    counts.index = labels
    retention.index = labels
    sizes.index = labels
    counts.to_csv(dirs["tables"] / "cohort_customer_counts.csv")
    retention.round(3).to_csv(dirs["tables"] / "cohort_retention_pct.csv")
 
    cust = orders.groupby("customer_unique_id").agg(
        total_orders=("order_id", "nunique"),
        total_spend=("order_value", "sum"),
        first_ts=("purchase_ts", "min"),
    )
    cust["repeat"] = (cust["total_orders"] >= 2).astype(int)
    return retention, sizes, cust
 
 
# ----------------------------------------------------------------
# 4. ML DATASET (first-purchase features only) + DEV / TEST SPLIT
# ----------------------------------------------------------------
# NOTE: purchase_month is intentionally NOT used (time proxy, <2 years of history).
NUM_FEATURES = [
    "order_value", "price_sum", "freight_sum", "freight_ratio", "item_count",
    "n_products", "n_sellers", "n_categories", "installments", "weight_sum_g",
    "photos_mean", "volume_mean", "same_state",
    "purchase_dow", "purchase_hour", "is_weekend",
]
CAT_FEATURES = ["customer_state", "seller_state", "category", "payment_type"]
 
 
def build_ml_dataset(orders, cutoff):
    first = orders.drop_duplicates("customer_unique_id", keep="first").copy()  # time-sorted
    first = first.rename(columns={"purchase_ts": "first_ts"})
 
    later = orders[["customer_unique_id", "purchase_ts"]].merge(
        first[["customer_unique_id", "first_ts"]], on="customer_unique_id")
    horizon = pd.Timedelta(days=FOLLOWUP_DAYS)
    in_window = later[(later["purchase_ts"] > later["first_ts"]) &
                      (later["purchase_ts"] <= later["first_ts"] + horizon)]
    repeat_ids = set(in_window["customer_unique_id"])
 
    # Only customers whose complete 90-day window is observable.
    first = first[first["first_ts"] + horizon <= cutoff].copy()
    first["target"] = first["customer_unique_id"].isin(repeat_ids).astype(int)
 
    first["purchase_dow"] = first["first_ts"].dt.dayofweek
    first["purchase_hour"] = first["first_ts"].dt.hour
    first["is_weekend"] = (first["purchase_dow"] >= 5).astype(int)
    for c in NUM_FEATURES:
        first[c] = pd.to_numeric(first[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in CAT_FEATURES:
        first[c] = first[c].astype("string").fillna("unknown").astype(str)
 
    ml = first[["customer_unique_id", "first_ts", "target"] + NUM_FEATURES + CAT_FEATURES]
    ml = ml.sort_values("first_ts").reset_index(drop=True)
    log.info("ML customers: %s | positive rate: %.2f%%", f"{len(ml):,}", ml["target"].mean() * 100)
    return ml
 
 
def split_dev_test(ml, test_days=90):
    """Dev (model selection + final fit) and ONE final Test.
 
    Test = last `test_days` of eligible first purchases.
    Dev  = customers whose 90-day label window closes before Test starts (purged).
    Only a 90-day band of customers is lost to the purge; no separate validation
    window is wasted, and the final model is refit on ALL of Dev.
    """
    horizon = pd.Timedelta(days=FOLLOWUP_DAYS)
    t_test = ml["first_ts"].max() - pd.Timedelta(days=test_days)
    test = ml[ml["first_ts"] > t_test].reset_index(drop=True)
    dev = ml[ml["first_ts"] + horizon <= t_test].reset_index(drop=True)
    purged = len(ml) - len(dev) - len(test)
    for name, part in [("dev", dev), ("test", test)]:
        if part.empty:
            raise SystemExit(f"{name} split is empty; dataset history is too short.")
        log.info("%-4s n=%6d  positives=%5d (%.2f%%)  %s -> %s", name, len(part),
                 part["target"].sum(), part["target"].mean() * 100,
                 part["first_ts"].min().date(), part["first_ts"].max().date())
        if part["target"].sum() < 30:
            log.warning("%s has very few positives; metrics will be noisy.", name)
    log.info("Purged band (label overlap with Test): %d customers", purged)
    return dev, test
 
 
# ----------------------------------------------------------------
# 5. MODELS
# ----------------------------------------------------------------
def build_preprocessor(scale):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:                                   # only linear models need scaling
        steps.append(("sc", StandardScaler()))
    num = Pipeline(steps)
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False)
    return ColumnTransformer([("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)])
 
 
def get_models():
    """name -> (estimator, search space, needs_scaling)"""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
            {"model__C": loguniform(1e-4, 1e2),
             "model__class_weight": [None, "balanced"]},
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(bootstrap=True, n_jobs=1, random_state=RANDOM_STATE),
            {"model__n_estimators": randint(100, 250),
             "model__max_depth": randint(4, 20),
             "model__min_samples_leaf": randint(5, 60),
             "model__max_features": uniform(0.2, 0.6),
             "model__max_samples": uniform(0.4, 0.5),
             "model__class_weight": ["balanced_subsample", None]},
            False,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(bootstrap=True, n_jobs=1, random_state=RANDOM_STATE),
            {"model__n_estimators": randint(100, 250),
             "model__max_depth": randint(4, 20),
             "model__min_samples_leaf": randint(5, 60),
             "model__max_features": uniform(0.2, 0.6),
             "model__max_samples": uniform(0.4, 0.5),
             "model__class_weight": ["balanced_subsample", None]},
            False,
        ),
        "Hist Gradient Boosting": (
            # early_stopping off: its internal validation split is random, not chronological
            HistGradientBoostingClassifier(early_stopping=False, random_state=RANDOM_STATE),
            {"model__learning_rate": loguniform(0.01, 0.2),
             "model__max_iter": randint(80, 300),
             "model__max_leaf_nodes": randint(8, 64),
             "model__max_depth": randint(3, 12),
             "model__min_samples_leaf": randint(20, 200),
             "model__l2_regularization": loguniform(1e-3, 10),
             "model__class_weight": [None, "balanced"]},
            False,
        ),
    }
    if HAS_LGBM:
        models["LightGBM"] = (
            LGBMClassifier(n_jobs=1, verbose=-1, subsample_freq=1, random_state=RANDOM_STATE),
            {"model__n_estimators": randint(100, 400),
             "model__learning_rate": loguniform(0.01, 0.2),
             "model__num_leaves": randint(8, 64),
             "model__min_child_samples": randint(20, 200),
             "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.5, 0.5),
             "model__reg_lambda": loguniform(1e-3, 10),
             "model__scale_pos_weight": loguniform(1, 30)},
            False,
        )
    else:  # fallback keeps the "5 models" requirement without extra installs
        models["Gradient Boosting"] = (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": randint(60, 200),
             "model__learning_rate": loguniform(0.02, 0.3),
             "model__max_depth": randint(2, 5),
             "model__min_samples_leaf": randint(20, 150),
             "model__subsample": uniform(0.6, 0.4)},
            False,
        )
    return models
 
 
# ---------- metrics / helpers ----------
def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))]) if len(thr) else 0.5
 
 
def topk_metrics(y, p, frac=TOP_FRAC):
    y = np.asarray(y)
    k = max(1, int(np.ceil(len(y) * frac)))
    idx = np.argsort(-p)[:k]
    prec_k = y[idx].mean()
    base = y.mean()
    return {"precision_top10": prec_k,
            "recall_top10": y[idx].sum() / max(y.sum(), 1),
            "lift_top10": prec_k / base if base > 0 else np.nan}
 
 
def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    base = float(np.mean(y))
    out = {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "brier_baseline": base * (1 - base),          # constant "base-rate" predictor
        "threshold": thr,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "flagged_share": float(pred.mean()),
    }
    out.update(topk_metrics(y, p))
    return out
 
 
def bootstrap_ci(y, p, n=300):
    """95% bootstrap interval on the ranking metrics (Test has few positives)."""
    rng = np.random.RandomState(RANDOM_STATE)
    vals = {"pr_auc": [], "roc_auc": [], "lift_top10": []}
    y = np.asarray(y)
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        vals["pr_auc"].append(average_precision_score(y[i], p[i]))
        vals["roc_auc"].append(roc_auc_score(y[i], p[i]))
        vals["lift_top10"].append(topk_metrics(y[i], p[i])["lift_top10"])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}
 
 
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)
 
 
def fit_platt(p_oof, y):
    """Sigmoid calibration learned on out-of-fold scores (ranking is unchanged)."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_oof), y)
 
 
def apply_platt(cal, p):
    return cal.predict_proba(_logit(p))[:, 1]
 
 
class PurgedTimeSplit:
    """Expanding-window time CV with a purge gap measured in DAYS (not rows).
 
    A training customer is used for a fold only if their 90-day label window
    closes before the fold's test period starts. Test folds cover the later
    (1 - start_frac) of Dev; the earliest part is only ever used for training.
    """
 
    def __init__(self, times, n_splits=CV_FOLDS, gap_days=FOLLOWUP_DAYS,
                 start_frac=0.5, min_train=1500):
        t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
        n = len(t)
        edges = [int(n * f) for f in np.linspace(start_frac, 1.0, n_splits + 1)]
        edges[-1] = n
        gap = pd.Timedelta(days=gap_days)
        self.folds = []
        for i in range(n_splits):
            test_idx = np.arange(edges[i], edges[i + 1])
            start = t.iloc[edges[i]]
            train_idx = np.where(t.values + gap.to_timedelta64() <= start.to_datetime64())[0]
            if len(train_idx) >= min_train and len(test_idx) > 0:
                self.folds.append((train_idx, test_idx))
        if len(self.folds) < 2:
            raise SystemExit("Not enough history for purged CV folds.")
 
    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)
 
    def split(self, X=None, y=None, groups=None):
        for tr, te in self.folds:
            yield tr, te
 
 
def _fit_predict(est, X_tr, y_tr, X_te):
    m = clone(est).fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]
 
 
def oof_predictions(est, X, y, cv):
    """Out-of-fold probabilities for the later part of Dev (fold models run in parallel)."""
    outs = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_predict)(est, X.iloc[tr], y[tr], X.iloc[te]) for tr, te in cv.folds)
    oof = np.full(len(y), np.nan)
    scores = []
    for (tr, te), p in zip(cv.folds, outs):
        oof[te] = p
        scores.append(average_precision_score(y[te], p) if y[te].sum() > 0 else np.nan)
    return oof, scores
 
 
def train_models(dev, test, dirs, trials):
    feats = NUM_FEATURES + CAT_FEATURES
    X_dev, y_dev = dev[feats].reset_index(drop=True), dev["target"].to_numpy()
    X_te, y_te = test[feats], test["target"].to_numpy()
 
    cv = PurgedTimeSplit(dev["first_ts"].values)
    for i, (a, b) in enumerate(cv.folds, 1):
        log.info("CV fold %d: train=%d (pos %d) -> test=%d (pos %d)", i, len(a),
                 y_dev[a].sum(), len(b), y_dev[b].sum())
 
    cache_dir = dirs["models"] / "_cache"
    memory = Memory(str(cache_dir), verbose=0)   # preprocessing fitted once per fold, not per trial
 
    rows, best_pipes, oof_all, search_rows = [], {}, {}, []
    for name, (est, space, scale) in get_models().items():
        log.info(">>> %s: %d trials x %d folds", name, trials, len(cv.folds))
        pipe = Pipeline([("prep", build_preprocessor(scale)), ("model", est)], memory=memory)
        search = RandomizedSearchCV(
            pipe, space, n_iter=trials, scoring="average_precision", cv=cv,
            n_jobs=N_JOBS, random_state=RANDOM_STATE, refit=False,
            error_score=np.nan, verbose=0)
        t0 = time.time()
        search.fit(X_dev, y_dev)
        secs = time.time() - t0
 
        failed = int(np.isnan(search.cv_results_["mean_test_score"]).sum())
        if failed:
            log.warning("%s: %d of %d trials FAILED in CV (check parameters/versions)",
                        name, failed, trials)
 
        best = Pipeline([("prep", build_preprocessor(scale)), ("model", clone(est))])
        best.set_params(**search.best_params_)                    # no cache: used for OOF + final fit
        oof, fold_sc = oof_predictions(best, X_dev, y_dev, cv)
        mask = ~np.isnan(oof)
 
        # accuracy / precision / recall / F1 for EVERY model, from out-of-fold data only
        # (Platt-calibrated, threshold = best F1 on the same OOF pool -> slightly optimistic)
        y_m, p_m = y_dev[mask], oof[mask]
        p_m_cal = apply_platt(fit_platt(p_m, y_m), p_m)
        mm = evaluate(y_m, p_m_cal, best_threshold(y_m, p_m_cal))
        rows.append({"model": name,
                     "cv_pr_auc_mean": float(np.nanmean(fold_sc)),
                     "cv_pr_auc_std": float(np.nanstd(fold_sc)),
                     "oof_pr_auc_pooled": average_precision_score(y_dev[mask], oof[mask]),
                     "oof_roc_auc": roc_auc_score(y_dev[mask], oof[mask]),
                     **{f"oof_{k}": mm[k] for k in ["accuracy", "precision", "recall", "f1",
                                                    "precision_top10", "recall_top10", "lift_top10"]},
                     "failed_trials": failed, "search_seconds": secs})
        search_rows.append({"model": name,
                            "best_params": json.dumps(search.best_params_, default=float)})
        best_pipes[name], oof_all[name] = best, oof
        pd.DataFrame(search.cv_results_).to_csv(
            dirs["tables"] / f"cv_results_{name.lower().replace(' ', '_')}.csv", index=False)
        log.info("%s | CV PR-AUC %.4f +- %.4f | OOF acc %.4f prec %.4f rec %.4f F1 %.4f | %.0fs", name,
                 rows[-1]["cv_pr_auc_mean"], rows[-1]["cv_pr_auc_std"], mm["accuracy"],
                 mm["precision"], mm["recall"], mm["f1"], secs)
 
    shutil.rmtree(cache_dir, ignore_errors=True)
 
    results = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)
    results.to_csv(dirs["tables"] / "model_comparison_cv.csv", index=False)
    pd.DataFrame(search_rows).to_csv(dirs["tables"] / "best_hyperparameters.csv", index=False)
 
    winner = results.loc[0, "model"]                       # chosen on CV only; Test untouched so far
    gap = results.loc[0, "cv_pr_auc_mean"] - results.loc[1, "cv_pr_auc_mean"] if len(results) > 1 else np.inf
    if gap < results.loc[0, "cv_pr_auc_std"]:
        log.warning("Winner leads runner-up by less than 1 fold-std: model differences are within noise.")
    log.info("Winner (by CV PR-AUC): %s", winner)
 
    # --- calibration + threshold from pooled out-of-fold scores of the winner ---
    mask = ~np.isnan(oof_all[winner])
    y_o, p_o = y_dev[mask], oof_all[winner][mask]
    cal = fit_platt(p_o, y_o)
    p_o_cal = apply_platt(cal, p_o)
    thr_f1 = best_threshold(y_o, p_o_cal)
    top_cut = float(np.quantile(p_o_cal, 1 - TOP_FRAC))
 
    # --- final model: refit on ALL Dev, evaluate on Test exactly once ---
    final = clone(best_pipes[winner]).fit(X_dev, y_dev)
    p_te_raw = final.predict_proba(X_te)[:, 1]
    p_te = apply_platt(cal, p_te_raw)
    test_metrics = evaluate(y_te, p_te, thr_f1)
    ci = bootstrap_ci(y_te, p_te)
 
    # --- permutation importance on a held-out CV fold (not on Test) ---
    tr, te = cv.folds[-1]
    m_last = clone(best_pipes[winner]).fit(X_dev.iloc[tr], y_dev[tr])
    pi = permutation_importance(m_last, X_dev.iloc[te], y_dev[te], scoring="average_precision",
                                n_repeats=5, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp.to_csv(dirs["tables"] / "permutation_importance.csv", index=False)
 
    joblib.dump({"pipeline": final, "calibrator": cal, "threshold": thr_f1,
                 "top10_cutoff": top_cut, "features": feats},
                dirs["models"] / "best_retention_model.joblib")
    pd.DataFrame({"customer_unique_id": test["customer_unique_id"].values,
                  "first_purchase": test["first_ts"].values, "actual": y_te,
                  "score_raw": p_te_raw, "repeat_probability": p_te}
                 ).to_csv(dirs["tables"] / "test_predictions_best_model.csv", index=False)
    pd.DataFrame([{**test_metrics, **{f"{k}_ci_low": v[0] for k, v in ci.items()},
                   **{f"{k}_ci_high": v[1] for k, v in ci.items()}}]
                 ).to_csv(dirs["tables"] / "test_metrics_winner.csv", index=False)
 
    art = {"oof_all": oof_all, "mask": mask, "y_dev": y_dev, "p_te": p_te, "y_te": y_te,
           "test_metrics": test_metrics, "ci": ci, "imp": imp, "thr": thr_f1, "top_cut": top_cut}
    return results, winner, art
 
 
# ----------------------------------------------------------------
# 6. THE TWO PLOTS
# ----------------------------------------------------------------
def plot_overview(retention, sizes, cust, orders, dirs, last_full_p):
    keep = sizes[sizes >= 300].index                      # drop tiny 2016 cohorts
    ret = retention.loc[keep].iloc[:, 1:10]               # months 1..9 (month 0 is always 100%)
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
 
    ax = fig.add_subplot(gs[:, 0])
    sns.heatmap(ret, annot=True, fmt=".2f", cmap="Blues", mask=ret.isna(), ax=ax,
                cbar_kws={"label": "Customers active again (%)"})
    ax.set_title("Cohort retention: % of cohort that buys again, by months since first purchase")
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Acquisition cohort (cohorts with >=300 customers)")
 
    ax = fig.add_subplot(gs[0, 1])
    label = cust["repeat"].map({0: "One-time", 1: "Repeat"})
    share = label.value_counts(normalize=True) * 100
    cnt = label.value_counts()
    sns.barplot(x=share.index, y=share.values, ax=ax, hue=share.index, legend=False)
    for i, k in enumerate(share.index):
        ax.text(i, share[k] + 1, f"{share[k]:.1f}%\n({cnt[k]:,})", ha="center")
    ax.set_ylim(0, 115)
    ax.set_title("One-time vs repeat customers (ever; recent cohorts had less time)")
    ax.set_xlabel("")
    ax.set_ylabel("Customers (%)")
 
    ax = fig.add_subplot(gs[1, 1])
    monthly = orders.groupby(orders["purchase_ts"].dt.to_period("M"))["order_id"].nunique()
    monthly = monthly[(monthly.index >= pd.Period("2017-01")) & (monthly.index <= last_full_p)]
    ax.plot(monthly.index.astype(str), monthly.values, marker="o")
    ax.set_title("Valid orders per month")
    ax.set_ylabel("Orders")
    ax.tick_params(axis="x", rotation=70)
 
    fig.suptitle("Plot 1 - Customer Retention Overview", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "01_cohort_and_customer_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_models(results, winner, art, dirs):
    oof_all, mask, y_dev = art["oof_all"], art["mask"], art["y_dev"]
    y_o = y_dev[mask]
    y_te, p_te, tm = art["y_te"], art["p_te"], art["test_metrics"]
    base_dev, base_te = y_o.mean(), y_te.mean()
 
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6])
    axes = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            axes[i, j] = fig.add_subplot(gs[i, j])
 
    # (a) CV comparison, mean +- std over folds; Test appears ONLY for the winner
    ax = axes[0, 0]
    r = results.sort_values("cv_pr_auc_mean").reset_index(drop=True)
    colors = ["tab:green" if m == winner else "tab:blue" for m in r["model"]]
    ax.barh(r["model"], r["cv_pr_auc_mean"], xerr=r["cv_pr_auc_std"], capsize=4, color=colors)
    ax.axvline(base_dev, color="red", ls="--", label=f"Random baseline ({base_dev:.3f})")
    ax.scatter([tm["pr_auc"]], [int(r.index[r["model"] == winner][0])], marker="*", s=300,
               color="black", zorder=5, label="Test PR-AUC (winner, used once)")
    ax.set_title("Purged time-CV PR-AUC (mean +- std over folds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
 
    # (b) PR curves from out-of-fold predictions
    ax = axes[0, 1]
    for name, o in oof_all.items():
        prec, rec, _ = precision_recall_curve(y_o, o[mask])
        ax.plot(rec, prec, label=f"{name} OOF (AP={average_precision_score(y_o, o[mask]):.3f})",
                lw=3 if name == winner else 1.3)
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    ax.plot(rec, prec, "k--", lw=2, label=f"{winner} TEST (AP={tm['pr_auc']:.3f})")
    ax.axhline(base_dev, color="red", ls=":", label="Random")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (out-of-fold + winner on Test)")
    ax.set_ylim(0, max(0.10, base_dev * 5))
    ax.legend(fontsize=8)
 
    # (c) cumulative gains on Test (winner)
    ax = axes[1, 0]
    order = np.argsort(-p_te)
    gains = np.cumsum(y_te[order]) / max(y_te.sum(), 1)
    frac = np.arange(1, len(p_te) + 1) / len(p_te)
    ax.plot(frac * 100, gains * 100, lw=2.5, label=winner)
    ax.plot([0, 100], [0, 100], "r--", label="Random targeting")
    t10 = gains[int(np.ceil(TOP_FRAC * len(p_te))) - 1] * 100
    ax.scatter([TOP_FRAC * 100], [t10], color="black", zorder=5)
    ax.annotate(f"Top 10% of customers\ncapture {t10:.0f}% of repeaters", (10, t10),
                xytext=(25, max(t10 - 15, 5)), arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("% of customers contacted (highest score first)")
    ax.set_ylabel("% of repeat buyers captured")
    ax.set_title(f"Cumulative gains on Test - {winner}")
    ax.legend()
 
    # (d) permutation importance on a held-out CV fold
    ax = axes[1, 1]
    top = art["imp"].head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set_title(f"Permutation importance (held-out CV fold) - {winner}")
    ax.set_xlabel("Mean decrease in PR-AUC")
 
    # (e) metrics table: every model (out-of-fold) + winner on Test
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    head = ["Model", "PR-AUC", "ROC-AUC", "Accuracy*", "Precision", "Recall", "F1"]
    cells = [[r.model, f"{r.oof_pr_auc_pooled:.4f}", f"{r.oof_roc_auc:.4f}", f"{r.oof_accuracy:.4f}",
              f"{r.oof_precision:.4f}", f"{r.oof_recall:.4f}", f"{r.oof_f1:.4f}"]
             for r in results.itertuples()]
    cells.append([f"TEST - {winner}", f"{tm['pr_auc']:.4f}", f"{tm['roc_auc']:.4f}",
                  f"{tm['accuracy']:.4f}", f"{tm['precision']:.4f}", f"{tm['recall']:.4f}",
                  f"{tm['f1']:.4f}"])
    tbl = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.9)
    for j in range(len(head)):
        tbl[(len(cells), j)].set_facecolor("#ffe9a8")          # Test row
    ax.set_title("Metrics at the F1-optimal threshold (models: out-of-fold; last row: Test, used once)\n"
                 "*Accuracy is misleading here: predicting 'nobody returns' already scores ~98%",
                 fontsize=11)
 
    fig.suptitle("Plot 2 - Model Evaluation (purged time-CV + one-time chronological Test)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "02_model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
# ----------------------------------------------------------------
# 7. WRITTEN REPORT
# ----------------------------------------------------------------
def _md_table(df, cols):
    t = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        t += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return t
 
 
def write_report(dirs, cust, retention, sizes, ml, dev, test, results, winner, art,
                 trials, cutoff, n_rows):
    tm, ci = art["test_metrics"], art["ci"]
    big = sizes[sizes >= 300].index
    m1 = retention.loc[big, 1].mean()
    cols = ["model", "cv_pr_auc_mean", "cv_pr_auc_std", "oof_pr_auc_pooled", "oof_roc_auc",
            "oof_accuracy", "oof_precision", "oof_recall", "oof_f1"]
    md_table = _md_table(results[cols].round(4), cols)
    top_feats = ", ".join(art["imp"].head(5)["feature"])
    small = "\n- **Dataset is small** (%s rows): few positives, so every metric is noisy." % f"{n_rows:,}" \
        if n_rows < 1_000_000 else ""
 
    text = f"""# Customer Retention Report
 
## 1. Data summary
- Rows: **{n_rows:,}** | Customers: **{len(cust):,}** | Ever-repeat customers: **{cust['repeat'].sum():,} ({cust['repeat'].mean()*100:.2f}%)**
- Data ends: {(cutoff - pd.Timedelta(days=1)).date()} (found from order density)
- Average month-1 retention across big cohorts: **{m1:.2f}%** (very low = most customers never return)
 
## 2. ML task
Predict whether a customer places another valid order within **{FOLLOWUP_DAYS} days** of the first order, using only information known at the first purchase.
- Eligible customers (full 90-day window observed): {len(ml):,}, 90-day positive rate {ml['target'].mean()*100:.2f}%
- Dev {len(dev):,} ({dev['target'].sum()} positives) | Test {len(test):,} ({test['target'].sum()} positives)
- Dev is purged: no Dev label window overlaps Test. Model selection used purged expanding-window CV inside Dev; the final model is refit on ALL Dev.
- {trials} randomised trials x 5 models, tuned on PR-AUC.
- Threshold and probability calibration (Platt) were learned from pooled out-of-fold predictions. Test was used once, for the winner only.
 
## 3. Model selection (CV on Dev, mean +- std over folds)
{md_table}
(oof_* = out-of-fold metrics at the F1-optimal threshold of each model; threshold is tuned on the same pool, so they are slightly optimistic. Accuracy is misleading with ~2% positives.)
 
**Selected model: {winner}** (best CV PR-AUC). If the gap to the runner-up is smaller than the fold std, the models are statistically indistinguishable.
 
## 4. Test result (winner only, 95% bootstrap CI)
- PR-AUC **{tm['pr_auc']:.4f}** [{ci['pr_auc'][0]:.4f}, {ci['pr_auc'][1]:.4f}] vs random baseline {test['target'].mean():.4f}
- ROC-AUC **{tm['roc_auc']:.4f}** [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}]
- Top 10% scored customers: precision {tm['precision_top10']:.3f}, captures {tm['recall_top10']*100:.1f}% of repeaters, lift **{tm['lift_top10']:.2f}x** [{ci['lift_top10'][0]:.2f}, {ci['lift_top10'][1]:.2f}]
- At the OOF-tuned threshold {tm['threshold']:.4f}: accuracy {tm['accuracy']:.4f}, precision {tm['precision']:.3f}, recall {tm['recall']:.3f}, F1 {tm['f1']:.3f}, flags {tm['flagged_share']*100:.1f}% of customers
- Brier (calibrated) {tm['brier']:.5f} vs base-rate predictor {tm['brier_baseline']:.5f}
- Most important features (held-out CV fold): {top_feats}
 
## 5. Honest limitations
- Repeat purchasing is rare, so absolute precision/F1 stay low; judge the model by PR-AUC and lift, not accuracy.{small}
- Test positives are few: use the confidence intervals, not the point estimates.
- Only first-order features are available (no browsing/marketing data), which caps achievable performance.
- Base rate can shift between Dev and Test (seasonality, growth); the F1 threshold may need re-tuning on fresh data.
- 'No return yet' is not permanent churn; the last months have less follow-up.
"""
    (dirs["reports"] / "retention_report.md").write_text(text, encoding="utf-8")
    log.info("Report saved: %s", dirs["reports"] / "retention_report.md")
 
 
# ----------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
 
    dirs = make_dirs(Path(args.out) if args.out else OUTPUT_DIR)
    t_all = time.time()
 
    raw = load_raw(find_dataset(args.data))
    n_rows = len(raw)
    orders = build_orders(raw)
    del raw
 
    cutoff = find_cutoff(orders)
    orders = orders[orders["purchase_ts"] < cutoff].reset_index(drop=True)   # drop stray late rows
    last_full_p = last_full_period(cutoff)
 
    retention, sizes, cust = cohort_analysis(orders, dirs, last_full_p)
    ml = build_ml_dataset(orders, cutoff)
    plot_overview(retention, sizes, cust, orders, dirs, last_full_p)          # PLOT 1
 
    dev, test = split_dev_test(ml, args.test_days)
    if min(dev["target"].nunique(), test["target"].nunique()) < 2:
        raise SystemExit("A split has a single target class; ML is not meaningful for this data.")
 
    results, winner, art = train_models(dev, test, dirs, args.trials)
    plot_models(results, winner, art, dirs)                                   # PLOT 2
    write_report(dirs, cust, retention, sizes, ml, dev, test, results, winner, art,
                 args.trials, cutoff, n_rows)
    log.info("DONE in %.1f min. Outputs: %s", (time.time() - t_all) / 60, dirs["reports"].parent)
 
 
if __name__ == "__main__":
    main()
 



