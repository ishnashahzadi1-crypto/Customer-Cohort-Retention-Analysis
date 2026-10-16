"""
CUSTOMER COHORT RETENTION + REPEAT-PURCHASE PREDICTION  (v3)
============================================================
Run:
    python Retention_v3.py                          (CSV in same folder / data folder)
    python Retention_v3.py --data path/to/file.csv
    python Retention_v3.py --quick                  (4 trials per model, ~fastest)
    python Retention_v3.py --trials 20              (slower, slightly better tuning)
    python Retention_v3.py --session-hours 0        (OLD target: any later order counts)
 
Outputs (./outputs/customer_retention):
    figures/eda/*.png   -> 17 separate EDA plots (understand the dataset)
    figures/ml/*.png    -> 13 separate ML plots  (understand the models)
    tables/*.csv        -> cohort tables, model comparison, test metrics, predictions
    models/best_retention_model.joblib
    reports/retention_report.md
 
WHAT WAS FIXED vs v2 (every weakness from the review)
  1. TARGET: ~45% of the old "90-day repeaters" re-ordered within 24 hours (split
     checkouts, not real loyalty). Orders in the first SESSION_HOURS (24h) now count as
     part of the first purchase. Prediction time = end of that session; label = another
     order in the following 90 days. (--session-hours 0 restores the old target.)
  2. FEATURES: added distance seller<->customer, promised delivery days, shipping limit,
     product name/description length, customer lat/lng, payment records, and the
     session features (orders/value in first 24h). Category = most expensive item.
  3. DRIFT: platform ramp-up (2016) excluded (--min-date); positive rate by month is
     plotted (eda/10) so the Dev/Test shift is visible, not hidden.
  4. OPERATING POINT: threshold = best F1 under a contact budget (--max-flag, default 5%
     of customers). This keeps accuracy/precision meaningful instead of flagging 20%.
  5. MODELS: LR, RF, ExtraTrees, HistGB, LightGBM (if installed) + soft-voting Ensemble
     (average of Platt-calibrated model scores). Winner = best purged-CV PR-AUC.
  6. NOISE: 95% bootstrap CI for every model's out-of-fold PR-AUC and for Test metrics.
  7. DATA END: found from order density (50% of typical daily orders), so the thin
     tail of the export is not treated as complete follow-up.
  8. SPEED: small regularised search spaces, no sklearn GradientBoosting fallback,
     no joblib cache, parallel search + parallel fold fits.
  9. Test set is used ONCE (winner only). scikit-learn >= 1.2 is checked at start.
"""
 
import argparse
import json
import logging
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
import sklearn
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, uniform
 
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
 
if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < (1, 2):
    raise SystemExit("scikit-learn >= 1.2 is required (pip install -U scikit-learn)")
 
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:  # LightGBM is optional
    HAS_LGBM = False
 
# ----------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------
RANDOM_STATE = 42
SESSION_HOURS = 24        # orders in the first N hours = part of the first purchase
FOLLOWUP_DAYS = 90        # label window after the session
CV_FOLDS = 4
N_JOBS = -1
MAX_FLAG = 0.05           # contact budget: flag at most 5% of customers
TOP_FRAC = 0.10
DENSITY_RATIO = 0.5       # data end = last day with >= 50% of typical daily orders
FIG_DPI = 110
INVALID_STATUSES = {"canceled", "cancelled", "unavailable"}
 
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "customer_retention"
 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
sns.set_theme(style="whitegrid", context="notebook")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("retention")
 
NEEDED_COLS = [
    "order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value",
    "order_status", "order_purchase_timestamp", "order_estimated_delivery_date",
    "shipping_limit_date", "customer_unique_id", "customer_state",
    "product_category_name_english", "seller_state", "payment_total_value",
    "payment_types", "payment_installments_max", "payment_record_count",
    "product_weight_g", "product_photos_qty", "product_length_cm", "product_height_cm",
    "product_width_cm", "product_name_lenght", "product_description_lenght",
    "customer_geolocation_lat", "customer_geolocation_lng",
    "seller_geolocation_lat", "seller_geolocation_lng",
]
 
# only information known at the END OF THE FIRST SESSION is used
NUM_FEATURES = [
    "order_value", "price_sum", "freight_sum", "freight_ratio", "item_count",
    "n_products", "n_sellers", "n_categories", "installments", "payment_records",
    "weight_sum_g", "photos_mean", "volume_mean", "name_len_mean", "desc_len_mean",
    "distance_km", "promised_days", "ship_limit_days", "cust_lat", "cust_lng",
    "same_state", "purchase_dow", "purchase_hour", "is_weekend",
    "session_orders", "session_value",
]
CAT_FEATURES = ["customer_state", "seller_state", "category", "payment_type"]
 
 
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
    for p in [PROJECT_DIR / "olist_all_files_merged.csv",
              PROJECT_DIR / "data" / "olist_all_files_merged.csv",
              PROJECT_DIR.parent / "olist_all_files_merged.csv",
              Path("/mnt/user-data/uploads/olist_all_files_merged.csv")]:
        if p.is_file():
            return p
    raise FileNotFoundError("CSV not found. Use --data path/to/olist_all_files_merged.csv")
 
 
def savefig(fig, dirs, sub, name):
    d = dirs["figures"] / sub
    d.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(d / name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
 
 
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
 
 
def haversine_km(lat1, lng1, lat2, lng2):
    a1, b1, a2, b2 = map(np.radians, [lat1, lng1, lat2, lng2])
    x = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((b2 - b1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(x))
 
 
def build_orders(df):
    df = df.copy()
    for c in ["order_purchase_timestamp", "order_estimated_delivery_date", "shipping_limit_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["order_status"] = df["order_status"].astype("string").str.strip().str.lower()
    df = df.dropna(subset=["order_id", "customer_unique_id", "order_purchase_timestamp"])
    df = df[~df["order_status"].isin(INVALID_STATUSES)]
 
    before = len(df)
    df = df.drop_duplicates(["order_id", "order_item_id"])
    if len(df) != before:
        log.warning("Dropped %d duplicated (order_id, order_item_id) rows", before - len(df))
 
    num_cols = ["price", "freight_value", "payment_total_value", "payment_installments_max",
                "payment_record_count", "product_weight_g", "product_photos_qty",
                "product_length_cm", "product_height_cm", "product_width_cm",
                "product_name_lenght", "product_description_lenght", "customer_geolocation_lat",
                "customer_geolocation_lng", "seller_geolocation_lat", "seller_geolocation_lng"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["price"] < 0, "price"] = np.nan
    df.loc[df["freight_value"] < 0, "freight_value"] = np.nan
    df["volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    df["dist_km"] = haversine_km(df["customer_geolocation_lat"], df["customer_geolocation_lng"],
                                 df["seller_geolocation_lat"], df["seller_geolocation_lng"])
 
    for c in ["customer_state", "seller_state", "product_category_name_english", "payment_types"]:
        df[c] = df[c].astype("string").str.strip().fillna("unknown")
    df["_cat_known"] = df["product_category_name_english"].mask(
        df["product_category_name_english"] == "unknown")
 
    # most expensive item first -> "first" picks the item that best describes the order
    df = df.sort_values(["order_id", "price"], ascending=[True, False])
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
        payment_records=("payment_record_count", "first"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("_cat_known", "nunique"),
        weight_sum_g=("product_weight_g", "sum"),
        photos_mean=("product_photos_qty", "mean"),
        volume_mean=("volume_cm3", "mean"),
        name_len_mean=("product_name_lenght", "mean"),
        desc_len_mean=("product_description_lenght", "mean"),
        distance_km=("dist_km", "mean"),
        est_delivery=("order_estimated_delivery_date", "first"),
        ship_limit=("shipping_limit_date", "min"),
        cust_lat=("customer_geolocation_lat", "first"),
        cust_lng=("customer_geolocation_lng", "first"),
        category=("product_category_name_english", "first"),
        customer_state=("customer_state", "first"),
        seller_state=("seller_state", "first"),
        payment_type=("payment_types", "first"),
    ).reset_index()
 
    fallback = orders["price_sum"] + orders["freight_sum"]
    orders["order_value"] = orders["payment_total"].where(orders["payment_total"].notna(), fallback)
    orders["freight_ratio"] = orders["freight_sum"] / (orders["price_sum"] + orders["freight_sum"]).replace(0, np.nan)
    orders["same_state"] = (orders["customer_state"] == orders["seller_state"]).astype(int)
    orders["promised_days"] = ((orders["est_delivery"] - orders["purchase_ts"]).dt.total_seconds() / 86400)
    orders["ship_limit_days"] = ((orders["ship_limit"] - orders["purchase_ts"]).dt.total_seconds() / 86400)
    for c in ["promised_days", "ship_limit_days"]:
        orders.loc[orders[c] < 0, c] = np.nan
    orders["payment_type"] = orders["payment_type"].astype("string").str.split(",").str[0].str.strip()
    orders = orders.drop(columns=["est_delivery", "ship_limit"])
    orders = orders.sort_values(["purchase_ts", "order_id"]).reset_index(drop=True)
    log.info("Order table: %s orders, %s customers",
             f"{len(orders):,}", f"{orders['customer_unique_id'].nunique():,}")
    return orders
 
 
def find_cutoff(orders):
    """Exclusive end of the observed data, from order DENSITY (not the max timestamp)."""
    daily = orders.groupby(orders["purchase_ts"].dt.normalize())["order_id"].nunique()
    typical = daily.tail(120).median()
    busy = daily[daily >= max(1.0, DENSITY_RATIO * typical)]
    cutoff = busy.index.max() + pd.Timedelta(days=1)
    log.info("Data end (by order density): %s | orders after it dropped: %d",
             (cutoff - pd.Timedelta(days=1)).date(), int((orders["purchase_ts"] >= cutoff).sum()))
    return cutoff
 
 
def last_full_period(cutoff):
    last_day = cutoff - pd.Timedelta(days=1)
    p = pd.Period(last_day, "M")
    return p if last_day.day >= last_day.days_in_month - 3 else p - 1
 
 
# ----------------------------------------------------------------
# 3. COHORT ANALYSIS
# ----------------------------------------------------------------
def cohort_analysis(orders, dirs, last_full_p):
    o = orders[["customer_unique_id", "purchase_ts"]].copy()
    o["m"] = o["purchase_ts"].dt.year * 12 + o["purchase_ts"].dt.month
    o = o[o["m"] <= last_full_p.year * 12 + last_full_p.month].copy()
    o["cohort"] = o.groupby("customer_unique_id")["m"].transform("min")
    o["period"] = o["m"] - o["cohort"]
 
    act = o[["customer_unique_id", "cohort", "period"]].drop_duplicates()
    counts = act.groupby(["cohort", "period"])["customer_unique_id"].nunique().unstack(fill_value=0)
    sizes = counts[0]
    retention = counts.div(sizes, axis=0) * 100
    allowed = (o["m"].max() - counts.index.values)[:, None]
    retention = retention.where(retention.columns.values[None, :] <= allowed)
 
    labels = [f"{(c - 1) // 12}-{(c - 1) % 12 + 1:02d}" for c in counts.index]
    counts.index, retention.index, sizes.index = labels, labels, labels
    counts.to_csv(dirs["tables"] / "cohort_customer_counts.csv")
    retention.round(3).to_csv(dirs["tables"] / "cohort_retention_pct.csv")
 
    cust = orders.groupby("customer_unique_id").agg(
        total_orders=("order_id", "nunique"), total_spend=("order_value", "sum"),
        first_ts=("purchase_ts", "min"))
    cust["repeat"] = (cust["total_orders"] >= 2).astype(int)
    return retention, sizes, cust
 
 
# ----------------------------------------------------------------
# 4. ML DATASET + DEV / TEST SPLIT
# ----------------------------------------------------------------
def horizon():
    return pd.Timedelta(hours=SESSION_HOURS) + pd.Timedelta(days=FOLLOWUP_DAYS)
 
 
def build_ml_dataset(orders, cutoff, min_date):
    sess = pd.Timedelta(hours=SESSION_HOURS)
    first = orders.drop_duplicates("customer_unique_id", keep="first").copy()
    first = first.rename(columns={"purchase_ts": "first_ts"})
 
    tmp = orders[["customer_unique_id", "purchase_ts", "order_value"]].merge(
        first[["customer_unique_id", "first_ts"]], on="customer_unique_id")
    dt = tmp["purchase_ts"] - tmp["first_ts"]
    sess_agg = tmp[dt <= sess].groupby("customer_unique_id").agg(
        session_orders=("purchase_ts", "size"), session_value=("order_value", "sum"))
    new_ids = set(tmp.loc[(dt > sess) & (dt <= horizon()), "customer_unique_id"])
    old_ids = set(tmp.loc[(dt > pd.Timedelta(0)) & (dt <= pd.Timedelta(days=FOLLOWUP_DAYS)),
                          "customer_unique_id"])
 
    first = first.join(sess_agg, on="customer_unique_id")
    first = first[(first["first_ts"] + horizon() <= cutoff) &
                  (first["first_ts"] >= pd.Timestamp(min_date))].copy()
    first["target"] = first["customer_unique_id"].isin(new_ids).astype(int)
 
    ids = set(first["customer_unique_id"])
    stats = {"old_positives": len(ids & old_ids), "new_positives": int(first["target"].sum())}
    stats["removed_share"] = 1 - stats["new_positives"] / max(stats["old_positives"], 1)
 
    first["purchase_dow"] = first["first_ts"].dt.dayofweek
    first["purchase_hour"] = first["first_ts"].dt.hour
    first["is_weekend"] = (first["purchase_dow"] >= 5).astype(int)
    for c in NUM_FEATURES:
        first[c] = pd.to_numeric(first[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in CAT_FEATURES:
        first[c] = first[c].astype("string").fillna("unknown").astype(str)
 
    ml = first[["customer_unique_id", "first_ts", "target"] + NUM_FEATURES + CAT_FEATURES]
    ml = ml.sort_values("first_ts").reset_index(drop=True)
    log.info("ML customers: %s | positive rate: %.2f%% | old-definition positives: %d -> new: %d "
             "(%.0f%% were same-session orders)", f"{len(ml):,}", ml["target"].mean() * 100,
             stats["old_positives"], stats["new_positives"], stats["removed_share"] * 100)
    return ml, stats
 
 
def split_dev_test(ml, test_days=90):
    """Dev = customers whose label window closes before Test starts (purged); Test = last days."""
    t_test = ml["first_ts"].max() - pd.Timedelta(days=test_days)
    test = ml[ml["first_ts"] > t_test].reset_index(drop=True)
    dev = ml[ml["first_ts"] + horizon() <= t_test].reset_index(drop=True)
    for name, part in [("dev", dev), ("test", test)]:
        if part.empty:
            raise SystemExit(f"{name} split is empty; dataset history is too short.")
        log.info("%-4s n=%6d  positives=%5d (%.2f%%)  %s -> %s", name, len(part),
                 part["target"].sum(), part["target"].mean() * 100,
                 part["first_ts"].min().date(), part["first_ts"].max().date())
        if part["target"].sum() < 30:
            log.warning("%s has very few positives; metrics will be noisy.", name)
    return dev, test, t_test
 
 
# ----------------------------------------------------------------
# 5. MODELS
# ----------------------------------------------------------------
def build_preprocessor(scale):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:   # linear model: make skewed money/weight features roughly normal
        steps.append(("qt", QuantileTransformer(n_quantiles=200, output_distribution="normal",
                                                random_state=RANDOM_STATE)))
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False)
    return ColumnTransformer([("num", Pipeline(steps), NUM_FEATURES), ("cat", cat, CAT_FEATURES)])
 
 
def get_models():
    """name -> (estimator, search space, needs_scaling). Small, regularised spaces = fast + stable."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
            {"model__C": loguniform(1e-4, 1.0)}, True),
        "Random Forest": (
            RandomForestClassifier(bootstrap=True, n_jobs=1, random_state=RANDOM_STATE),
            {"model__n_estimators": randint(80, 160), "model__max_depth": randint(3, 9),
             "model__min_samples_leaf": randint(40, 200), "model__max_features": uniform(0.15, 0.3),
             "model__max_samples": uniform(0.3, 0.4),
             "model__class_weight": [None, "balanced_subsample"]}, False),
        "Extra Trees": (
            ExtraTreesClassifier(bootstrap=True, n_jobs=1, random_state=RANDOM_STATE),
            {"model__n_estimators": randint(80, 160), "model__max_depth": randint(3, 9),
             "model__min_samples_leaf": randint(40, 200), "model__max_features": uniform(0.15, 0.3),
             "model__max_samples": uniform(0.3, 0.4),
             "model__class_weight": [None, "balanced_subsample"]}, False),
        "Hist Gradient Boosting": (
            HistGradientBoostingClassifier(early_stopping=False, random_state=RANDOM_STATE),
            {"model__learning_rate": loguniform(0.02, 0.15), "model__max_iter": randint(60, 180),
             "model__max_leaf_nodes": randint(4, 24), "model__max_depth": randint(2, 6),
             "model__min_samples_leaf": randint(50, 300),
             "model__l2_regularization": loguniform(0.1, 30),
             "model__class_weight": [None, "balanced"]}, False),
    }
    if HAS_LGBM:
        models["LightGBM"] = (
            LGBMClassifier(n_jobs=1, verbose=-1, subsample_freq=1, random_state=RANDOM_STATE),
            {"model__n_estimators": randint(80, 300), "model__learning_rate": loguniform(0.02, 0.15),
             "model__num_leaves": randint(4, 24), "model__min_child_samples": randint(50, 300),
             "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.4, 0.6),
             "model__reg_lambda": loguniform(0.1, 30),
             "model__scale_pos_weight": loguniform(1, 10)}, False)
    return models
 
 
# ---------- metrics / helpers ----------
def best_threshold(y, p, max_flag=MAX_FLAG):
    """Best-F1 threshold among those that flag at most `max_flag` of customers."""
    y, p = np.asarray(y), np.asarray(p)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = (2 * prec * rec / np.clip(prec + rec, 1e-12, None))[:-1]
    ps = np.sort(p)
    share = 1 - np.searchsorted(ps, thr, side="left") / len(p)
    ok = share <= max_flag
    if not ok.any():
        return float(np.quantile(p, 1 - max_flag))
    idx = np.where(ok)[0]
    return float(thr[idx[int(np.argmax(f1[idx]))]])
 
 
def topk_metrics(y, p, frac=TOP_FRAC):
    y = np.asarray(y)
    k = max(1, int(np.ceil(len(y) * frac)))
    idx = np.argsort(-p)[:k]
    prec_k, base = y[idx].mean(), y.mean()
    return {"precision_top10": prec_k, "recall_top10": y[idx].sum() / max(y.sum(), 1),
            "lift_top10": prec_k / base if base > 0 else np.nan}
 
 
def evaluate(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    base = float(y.mean())
    out = {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)), "brier_baseline": base * (1 - base),
        "threshold": thr, "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0), "flagged_share": float(pred.mean()),
    }
    out.update(topk_metrics(y, p))
    return out
 
 
def bootstrap_ci(y, p, n=200):
    """95% bootstrap interval for the ranking metrics."""
    rng = np.random.RandomState(RANDOM_STATE)
    y, p = np.asarray(y), np.asarray(p)
    vals = {"pr_auc": [], "roc_auc": [], "lift_top10": []}
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        vals["pr_auc"].append(average_precision_score(y[i], p[i]))
        vals["roc_auc"].append(roc_auc_score(y[i], p[i]))
        vals["lift_top10"].append(topk_metrics(y[i], p[i])["lift_top10"])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}
 
 
def pr_ci(y, p, n=100):
    rng = np.random.RandomState(RANDOM_STATE)
    y, p = np.asarray(y), np.asarray(p)
    v = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum() > 0:
            v.append(average_precision_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
 
 
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)
 
 
def fit_platt(p, y):
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y)
 
 
def apply_platt(cal, p):
    return cal.predict_proba(_logit(p))[:, 1]
 
 
class PurgedTimeSplit:
    """Expanding-window time CV; a train customer is used only if their whole label
    window (session + 90 days) closes before the fold's test period starts."""
 
    def __init__(self, times, n_splits=CV_FOLDS, start_frac=0.5, min_train=1500):
        t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
        n = len(t)
        edges = [int(n * f) for f in np.linspace(start_frac, 1.0, n_splits + 1)]
        edges[-1] = n
        gap = horizon().to_timedelta64()
        self.folds = []
        for i in range(n_splits):
            te = np.arange(edges[i], edges[i + 1])
            start = t.iloc[edges[i]].to_datetime64()
            tr = np.where(t.values + gap <= start)[0]
            if len(tr) >= min_train and len(te) > 0:
                self.folds.append((tr, te))
        if len(self.folds) < 2:
            raise SystemExit("Not enough history for purged CV folds.")
 
    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)
 
    def split(self, X=None, y=None, groups=None):
        for tr, te in self.folds:
            yield tr, te
 
 
def _fit_predict(est, X_tr, y_tr, X_te):
    return clone(est).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
 
 
def _fit_full(est, X, y):
    return clone(est).fit(X, y)
 
 
def oof_predictions(est, X, y, cv):
    outs = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_predict)(est, X.iloc[tr], y[tr], X.iloc[te]) for tr, te in cv.folds)
    oof = np.full(len(y), np.nan)
    for (tr, te), p in zip(cv.folds, outs):
        oof[te] = p
    return oof
 
 
def fold_pr(y, p, cv):
    return [average_precision_score(y[te], p[te]) if y[te].sum() > 0 else np.nan for _, te in cv.folds]
 
 
def train_models(dev, test, dirs, trials):
    feats = NUM_FEATURES + CAT_FEATURES
    X_dev, y_dev = dev[feats].reset_index(drop=True), dev["target"].to_numpy()
    X_te, y_te = test[feats], test["target"].to_numpy()
 
    cv = PurgedTimeSplit(dev["first_ts"].values)
    for i, (a, b) in enumerate(cv.folds, 1):
        log.info("CV fold %d: train=%d (pos %d) -> test=%d (pos %d)", i, len(a),
                 y_dev[a].sum(), len(b), y_dev[b].sum())
    fold_base = [float(y_dev[te].mean()) for _, te in cv.folds]
 
    models = get_models()
    best_pipes, cals, oof_cal, fold_scores, meta, search_rows = {}, {}, {}, {}, {}, []
    for name, (est, space, scale) in models.items():
        pipe = Pipeline([("prep", build_preprocessor(scale)), ("model", est)])
        search = RandomizedSearchCV(
            pipe, space, n_iter=trials, scoring="average_precision", cv=cv, n_jobs=N_JOBS,
            random_state=RANDOM_STATE, refit=False, error_score=np.nan, verbose=0)
        t0 = time.time()
        search.fit(X_dev, y_dev)
        secs = time.time() - t0
        failed = int(np.isnan(search.cv_results_["mean_test_score"]).sum())
        if failed:
            log.warning("%s: %d of %d trials FAILED in CV", name, failed, len(search.cv_results_["params"]))
 
        best = Pipeline([("prep", build_preprocessor(scale)), ("model", clone(est))])
        best.set_params(**search.best_params_)
        oof = oof_predictions(best, X_dev, y_dev, cv)
        mask = ~np.isnan(oof)
        cals[name] = fit_platt(oof[mask], y_dev[mask])
        pc = np.full(len(y_dev), np.nan)
        pc[mask] = apply_platt(cals[name], oof[mask])
        best_pipes[name], oof_cal[name] = best, pc
        fold_scores[name] = fold_pr(y_dev, oof, cv)
        meta[name] = {"failed": failed, "secs": secs}
        search_rows.append({"model": name, "best_params": json.dumps(search.best_params_, default=float)})
        pd.DataFrame(search.cv_results_).to_csv(
            dirs["tables"] / f"cv_results_{name.lower().replace(' ', '_')}.csv", index=False)
        log.info("%-22s search %.0fs | CV PR-AUC %.4f +- %.4f", name, secs,
                 np.nanmean(fold_scores[name]), np.nanstd(fold_scores[name]))
 
    # ---- soft-voting ensemble of the calibrated out-of-fold scores ----
    names = list(models)
    mask = ~np.isnan(oof_cal[names[0]])
    ens = np.mean(np.vstack([oof_cal[n] for n in names]), axis=0)
    cals["Ensemble"] = fit_platt(ens[mask], y_dev[mask])
    pc = np.full(len(y_dev), np.nan)
    pc[mask] = apply_platt(cals["Ensemble"], ens[mask])
    oof_cal["Ensemble"], fold_scores["Ensemble"] = pc, fold_pr(y_dev, pc, cv)
    meta["Ensemble"] = {"failed": 0, "secs": 0.0}
 
    # ---- one summary row per model (out-of-fold, F1-best threshold under the contact budget) ----
    rows, thr_all = [], {}
    y_m = y_dev[mask]
    for name in oof_cal:
        p_m = oof_cal[name][mask]
        thr_all[name] = best_threshold(y_m, p_m)
        mm = evaluate(y_m, p_m, thr_all[name])
        lo, hi = pr_ci(y_m, p_m)
        rows.append({"model": name, "cv_pr_auc_mean": float(np.nanmean(fold_scores[name])),
                     "cv_pr_auc_std": float(np.nanstd(fold_scores[name])),
                     "oof_pr_auc_pooled": mm["pr_auc"], "oof_pr_auc_lo": lo, "oof_pr_auc_hi": hi,
                     "oof_roc_auc": mm["roc_auc"], "oof_accuracy": mm["accuracy"],
                     "oof_balanced_accuracy": mm["balanced_accuracy"],
                     "oof_precision": mm["precision"], "oof_recall": mm["recall"], "oof_f1": mm["f1"],
                     "oof_flagged_share": mm["flagged_share"],
                     "oof_precision_top10": mm["precision_top10"], "oof_lift_top10": mm["lift_top10"],
                     "threshold": thr_all[name], "failed_trials": meta[name]["failed"],
                     "search_seconds": meta[name]["secs"]})
        log.info("%-22s OOF acc %.3f | prec %.3f | rec %.3f | F1 %.3f | flagged %.1f%%", name,
                 mm["accuracy"], mm["precision"], mm["recall"], mm["f1"], mm["flagged_share"] * 100)
    results = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)
    results.to_csv(dirs["tables"] / "model_comparison_cv.csv", index=False)
    pd.DataFrame(search_rows).to_csv(dirs["tables"] / "best_hyperparameters.csv", index=False)
 
    winner = results.loc[0, "model"]
    if len(results) > 1 and (results.loc[0, "cv_pr_auc_mean"] - results.loc[1, "cv_pr_auc_mean"]
                             < results.loc[0, "cv_pr_auc_std"]):
        log.warning("Winner leads runner-up by less than 1 fold-std: differences are within noise.")
    log.info("Winner (by purged-CV PR-AUC): %s", winner)
 
    # ---- final fit on ALL Dev; Test is evaluated exactly once ----
    members = names if winner == "Ensemble" else [winner]
    fitted = dict(zip(members, Parallel(n_jobs=N_JOBS)(
        delayed(_fit_full)(best_pipes[m], X_dev, y_dev) for m in members)))
    pm = [apply_platt(cals[m], fitted[m].predict_proba(X_te)[:, 1]) for m in members]
    p_te = apply_platt(cals["Ensemble"], np.mean(pm, axis=0)) if winner == "Ensemble" else pm[0]
    thr = thr_all[winner]
    test_metrics = evaluate(y_te, p_te, thr)
    ci = bootstrap_ci(y_te, p_te)
 
    # ---- permutation importance of the best single model, on a held-out CV fold ----
    singles = results[results["model"] != "Ensemble"]
    fi_name = singles.loc[singles.index[0], "model"]
    tr, te = cv.folds[-1]
    te = te[:5000]
    m_last = clone(best_pipes[fi_name]).fit(X_dev.iloc[tr], y_dev[tr])
    pi = permutation_importance(m_last, X_dev.iloc[te], y_dev[te], scoring="average_precision",
                                n_repeats=5, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp.to_csv(dirs["tables"] / "permutation_importance.csv", index=False)
 
    joblib.dump({"members": [{"name": m, "pipeline": fitted[m], "calibrator": cals[m]} for m in members],
                 "ensemble_calibrator": cals["Ensemble"] if winner == "Ensemble" else None,
                 "threshold": thr, "features": feats, "max_flag": MAX_FLAG,
                 "session_hours": SESSION_HOURS, "followup_days": FOLLOWUP_DAYS},
                dirs["models"] / "best_retention_model.joblib")
    pd.DataFrame({"customer_unique_id": test["customer_unique_id"].values,
                  "first_purchase": test["first_ts"].values, "actual": y_te,
                  "repeat_probability": p_te, "flag": (p_te >= thr).astype(int)}
                 ).to_csv(dirs["tables"] / "test_predictions_best_model.csv", index=False)
    pd.DataFrame([{**test_metrics, **{f"{k}_ci_low": v[0] for k, v in ci.items()},
                   **{f"{k}_ci_high": v[1] for k, v in ci.items()}}]
                 ).to_csv(dirs["tables"] / "test_metrics_winner.csv", index=False)
 
    art = {"oof_cal": oof_cal, "mask": mask, "y_dev": y_dev, "p_te": p_te, "y_te": y_te,
           "test_metrics": test_metrics, "ci": ci, "imp": imp, "fi_name": fi_name, "thr": thr,
           "fold_scores": fold_scores, "fold_base": fold_base}
    return results, winner, art
 
 
def score_customers(model_path, X):
    """Use the saved model on new first-purchase rows (same feature columns)."""
    art = joblib.load(model_path)
    X = X[art["features"]]
    p = np.mean([apply_platt(m["calibrator"], m["pipeline"].predict_proba(X)[:, 1])
                 for m in art["members"]], axis=0)
    if art["ensemble_calibrator"] is not None:
        p = apply_platt(art["ensemble_calibrator"], p)
    return p, (p >= art["threshold"]).astype(int)
 
 
# ----------------------------------------------------------------
# 6A. EDA PLOTS (each saved as its own file)  -> figures/eda
# ----------------------------------------------------------------
def eda_cohort_heatmap(retention, sizes, dirs, min_cohort):
    keep = sizes[sizes >= min_cohort].index
    ret = retention.loc[keep].iloc[:, 1:10]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(ret, annot=True, fmt=".2f", cmap="Blues", mask=ret.isna(), ax=ax,
                cbar_kws={"label": "Customers active again (%)"})
    ax.set(title="Cohort retention: % of a cohort that buys again, by months since first purchase",
           xlabel="Months since first purchase", ylabel=f"Acquisition cohort (>= {min_cohort} customers)")
    savefig(fig, dirs, "eda", "01_cohort_retention_heatmap.png")
 
 
def eda_orders_trend(orders, cust, dirs, last_full_p):
    m = orders.groupby(orders["purchase_ts"].dt.to_period("M"))["order_id"].nunique()
    n = cust["first_ts"].dt.to_period("M").value_counts().sort_index()
    lo = pd.Period("2017-01")
    m, n = m[(m.index >= lo) & (m.index <= last_full_p)], n[(n.index >= lo) & (n.index <= last_full_p)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(m.index.astype(str), m.values, marker="o", label="Orders")
    ax.plot(n.index.astype(str), n.reindex(m.index).values, marker="s", label="New customers")
    ax.set(title="Orders and new customers per month (complete months only)", ylabel="Count")
    ax.tick_params(axis="x", rotation=70)
    ax.legend()
    savefig(fig, dirs, "eda", "02_orders_and_new_customers.png")
 
 
def eda_orders_per_customer(cust, dirs):
    s = cust["total_orders"].clip(upper=5).value_counts().sort_index()
    labels = [str(i) if i < 5 else "5+" for i in s.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, s.values)
    ax.set_yscale("log")
    for i, v in enumerate(s.values):
        ax.text(i, v, f"{v:,}\n({v / s.sum() * 100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set(title="How many orders does a customer place? (log scale)",
           xlabel="Orders per customer", ylabel="Customers")
    ax.set_ylim(top=s.max() * 6)
    savefig(fig, dirs, "eda", "03_orders_per_customer.png")
 
 
def eda_days_to_second_order(orders, dirs):
    o = orders[["customer_unique_id", "purchase_ts"]].copy()
    o["rk"] = o.groupby("customer_unique_id").cumcount()
    a = o[o["rk"] == 0].set_index("customer_unique_id")["purchase_ts"]
    b = o[o["rk"] == 1].set_index("customer_unique_id")["purchase_ts"]
    d = (b - a.loc[b.index]).dt.total_seconds() / 86400
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(d[d <= 120], bins=np.arange(0, 121, 1), color="tab:blue")
    ax.axvspan(0, SESSION_HOURS / 24 if SESSION_HOURS else 1 / 24, color="red", alpha=0.25,
               label=f"first {SESSION_HOURS}h (counted as ONE purchase)" if SESSION_HOURS else "first hour")
    share = (d <= 1).mean() * 100
    ax.set(title=f"Days until the 2nd order ({share:.0f}% of repeaters re-order within 24h)",
           xlabel="Days after first order", ylabel="Customers")
    ax.set_yscale("log")
    ax.legend()
    savefig(fig, dirs, "eda", "04_days_to_second_order.png")
 
 
def eda_rate_by_bins(ml, col, title, fname, dirs, q=10):
    d = ml[[col, "target"]].dropna()
    bins = pd.qcut(d[col], q, duplicates="drop")
    g = d.groupby(bins, observed=True)["target"].agg(["mean", "size"])
    labels = [f"{iv.left:.3g}\n{iv.right:.3g}" for iv in g.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(g)), g["mean"] * 100)
    ax.axhline(d["target"].mean() * 100, color="red", ls="--", label="Overall rate")
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m * 100, f"n={n:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(title=title, xlabel=f"{col} (quantile bins: from / to)", ylabel="Repeat rate (%)")
    ax.legend()
    savefig(fig, dirs, "eda", fname)
 
 
def eda_rate_by_cat(ml, col, title, fname, dirs, top=15, min_n=200):
    g = ml.groupby(col)["target"].agg(["mean", "size"])
    g = g[g["size"] >= min_n].sort_values("size", ascending=False).head(top).sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(g) + 1.5)))
    ax.barh(g.index.astype(str), g["mean"] * 100)
    ax.axvline(ml["target"].mean() * 100, color="red", ls="--", label="Overall rate")
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(m * 100, i, f"  n={n:,}", va="center", fontsize=8)
    ax.set(title=title, xlabel="Repeat rate (%)")
    ax.legend()
    savefig(fig, dirs, "eda", fname)
 
 
def eda_monthly_rate(ml, t_test, dirs):
    m = ml.groupby(ml["first_ts"].dt.to_period("M"))["target"].agg(["mean", "size"])
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, m["size"], alpha=0.3, label="Customers")
    ax2 = ax.twinx()
    ax2.plot(x, m["mean"] * 100, color="tab:red", marker="o", label="Repeat rate (%)")
    ax2.grid(False)
    pos = list(m.index).index(pd.Period(t_test, "M")) if pd.Period(t_test, "M") in m.index else None
    if pos is not None:
        ax.axvline(pos - 0.5, color="black", ls="--")
        ax.text(pos - 0.4, m["size"].max() * 0.95, "Test starts", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(m.index.astype(str), rotation=70)
    ax.set(title="Repeat rate by first-purchase month (drift check)", ylabel="Customers")
    ax2.set_ylabel("Repeat rate (%)")
    savefig(fig, dirs, "eda", "10_repeat_rate_by_month.png")
 
 
def eda_correlation(ml, dirs):
    c = ml[NUM_FEATURES + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(c, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation between numeric features and the target")
    savefig(fig, dirs, "eda", "11_feature_correlation_heatmap.png")
    t = c["target"].drop("target").sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(t.index, t.values, color=np.where(t.values >= 0, "tab:green", "tab:red"))
    ax.set(title="Correlation of each numeric feature with repeat purchase", xlabel="Pearson r")
    savefig(fig, dirs, "eda", "12_correlation_with_target.png")
 
 
def eda_distributions(ml, dirs):
    cols = ["order_value", "freight_sum", "weight_sum_g", "distance_km", "promised_days", "photos_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), cols):
        v = ml[c].dropna()
        v = v[v <= v.quantile(0.99)]
        ax.hist(v, bins=50, color="tab:blue")
        ax.set_title(c)
    fig.suptitle("Distributions of key numeric features (clipped at the 99th percentile)", fontweight="bold")
    savefig(fig, dirs, "eda", "13_feature_distributions.png")
 
 
def eda_missing(ml, dirs):
    miss = ml[NUM_FEATURES + CAT_FEATURES].isna().mean().mul(100)
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(miss) + 1.5)))
    if len(miss):
        ax.barh(miss.index, miss.values)
    ax.set(title="Missing values per feature (%) - handled by median imputation + indicator",
           xlabel="% missing")
    savefig(fig, dirs, "eda", "14_missing_values.png")
 
 
# ----------------------------------------------------------------
# 6B. ML PLOTS (each saved as its own file)  -> figures/ml
# ----------------------------------------------------------------
def _colors(results, winner):
    return ["tab:green" if m == winner else "tab:blue" for m in results["model"]]
 
 
def ml_cv_comparison(results, winner, art, dirs):
    r = results.sort_values("cv_pr_auc_mean").reset_index(drop=True)
    base = art["y_dev"][art["mask"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(r["model"], r["cv_pr_auc_mean"], xerr=r["cv_pr_auc_std"], capsize=4, color=_colors(r, winner))
    ax.axvline(base, color="red", ls="--", label=f"Random baseline ({base:.3f})")
    ax.scatter([art["test_metrics"]["pr_auc"]], [int(r.index[r["model"] == winner][0])], marker="*", s=300,
               color="black", zorder=5, label="Test PR-AUC (winner, used once)")
    ax.set(title="Purged time-CV PR-AUC (mean +- std over folds)", xlabel="PR-AUC")
    ax.legend()
    savefig(fig, dirs, "ml", "01_model_cv_pr_auc.png")
 
 
def ml_metrics_bars(results, winner, art, dirs):
    tm = art["test_metrics"]
    spec = [("oof_pr_auc_pooled", "pr_auc", "PR-AUC"), ("oof_roc_auc", "roc_auc", "ROC-AUC"),
            ("oof_accuracy", "accuracy", "Accuracy"), ("oof_balanced_accuracy", "balanced_accuracy", "Balanced accuracy"),
            ("oof_precision", "precision", "Precision"), ("oof_recall", "recall", "Recall"),
            ("oof_f1", "f1", "F1")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    labels = list(results["model"]) + ["TEST\n" + winner]
    colors = _colors(results, winner) + ["gold"]
    for ax, (oc, tc, title) in zip(axes.ravel(), spec):
        vals = list(results[oc]) + [tm[tc]]
        ax.bar(range(len(vals)), vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(title)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Model metrics (out-of-fold at the contact-budget threshold; gold bar = Test, used once)",
                 fontweight="bold")
    savefig(fig, dirs, "ml", "02_metrics_comparison.png")
 
 
def ml_metrics_table(results, winner, art, dirs):
    tm = art["test_metrics"]
    head = ["Model", "PR-AUC (95% CI)", "ROC-AUC", "Accuracy", "Bal. Acc.", "Precision", "Recall", "F1", "Flagged %"]
    cells = [[r.model, f"{r.oof_pr_auc_pooled:.4f} [{r.oof_pr_auc_lo:.3f}, {r.oof_pr_auc_hi:.3f}]",
              f"{r.oof_roc_auc:.4f}", f"{r.oof_accuracy:.4f}", f"{r.oof_balanced_accuracy:.4f}",
              f"{r.oof_precision:.4f}", f"{r.oof_recall:.4f}", f"{r.oof_f1:.4f}",
              f"{r.oof_flagged_share * 100:.1f}"] for r in results.itertuples()]
    ci = art["ci"]["pr_auc"]
    cells.append([f"TEST - {winner}", f"{tm['pr_auc']:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]", f"{tm['roc_auc']:.4f}",
                  f"{tm['accuracy']:.4f}", f"{tm['balanced_accuracy']:.4f}", f"{tm['precision']:.4f}",
                  f"{tm['recall']:.4f}", f"{tm['f1']:.4f}", f"{tm['flagged_share'] * 100:.1f}"])
    fig, ax = plt.subplots(figsize=(17, 1.2 + 0.55 * len(cells)))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    for j in range(len(head)):
        tbl[(len(cells), j)].set_facecolor("#ffe9a8")
        tbl[(0, j)].set_facecolor("#dde6f5")
    ax.set_title(f"All metrics (models: out-of-fold; last row: Test) | threshold = best F1 when flagging <= {MAX_FLAG:.0%} of customers",
                 fontsize=12)
    savefig(fig, dirs, "ml", "03_metrics_table.png")
 
 
def ml_pr_curves(results, winner, art, dirs):
    m, y = art["mask"], art["y_dev"][art["mask"]]
    base = y.mean()
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, p in art["oof_cal"].items():
        pr, rc, _ = precision_recall_curve(y, p[m])
        ax.plot(rc, pr, lw=3 if name == winner else 1.3,
                label=f"{name} (AP={average_precision_score(y, p[m]):.3f})")
    pr, rc, _ = precision_recall_curve(art["y_te"], art["p_te"])
    ax.plot(rc, pr, "k--", lw=2, label=f"{winner} TEST (AP={art['test_metrics']['pr_auc']:.3f})")
    ax.axhline(base, color="red", ls=":", label=f"Random ({base:.3f})")
    ax.set(title="Precision-Recall curves (out-of-fold + winner on Test)", xlabel="Recall",
           ylabel="Precision", ylim=(0, max(0.10, base * 5)))
    ax.legend(fontsize=8)
    savefig(fig, dirs, "ml", "04_precision_recall_curves.png")
 
 
def ml_roc_curves(results, winner, art, dirs):
    m, y = art["mask"], art["y_dev"][art["mask"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, p in art["oof_cal"].items():
        fpr, tpr, _ = roc_curve(y, p[m])
        ax.plot(fpr, tpr, lw=3 if name == winner else 1.3, label=f"{name} (AUC={roc_auc_score(y, p[m]):.3f})")
    fpr, tpr, _ = roc_curve(art["y_te"], art["p_te"])
    ax.plot(fpr, tpr, "k--", lw=2, label=f"{winner} TEST (AUC={art['test_metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "r:", label="Random")
    ax.set(title="ROC curves (out-of-fold + winner on Test)", xlabel="False positive rate",
           ylabel="True positive rate")
    ax.legend(fontsize=8)
    savefig(fig, dirs, "ml", "05_roc_curves.png")
 
 
def ml_confusion(winner, art, dirs):
    y, p = art["y_te"], art["p_te"]
    cm = confusion_matrix(y, (p >= art["thr"]).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, t in [(axes[0], cm, "d", "Counts"),
                             (axes[1], cm / cm.sum(axis=1, keepdims=True), ".1%", "Row-normalised (recall view)")]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No repeat", "Repeat"], yticklabels=["No repeat", "Repeat"])
        ax.set(title=t, xlabel="Predicted", ylabel="Actual")
    fig.suptitle(f"Confusion matrix on Test - {winner} (threshold {art['thr']:.4f})", fontweight="bold")
    savefig(fig, dirs, "ml", "06_confusion_matrix_test.png")
 
 
def ml_gains(winner, art, dirs):
    y, p = art["y_te"], art["p_te"]
    order = np.argsort(-p)
    gains = np.cumsum(y[order]) / max(y.sum(), 1)
    frac = np.arange(1, len(p) + 1) / len(p)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(frac * 100, gains * 100, lw=2.5, label=winner)
    ax.plot([0, 100], [0, 100], "r--", label="Random targeting")
    t10 = gains[int(np.ceil(TOP_FRAC * len(p))) - 1] * 100
    ax.scatter([TOP_FRAC * 100], [t10], color="black", zorder=5)
    ax.annotate(f"Top 10% of customers\ncapture {t10:.0f}% of repeaters", (10, t10),
                xytext=(25, max(t10 - 15, 5)), arrowprops=dict(arrowstyle="->"))
    ax.set(title=f"Cumulative gains on Test - {winner}", xlabel="% of customers contacted (highest score first)",
           ylabel="% of repeat buyers captured")
    ax.legend()
    savefig(fig, dirs, "ml", "07_cumulative_gains_test.png")
 
 
def ml_lift(winner, art, dirs):
    y, p = art["y_te"], art["p_te"]
    chunks = np.array_split(np.argsort(-p), 10)
    base = y.mean()
    lift = [y[c].mean() / base if base > 0 else np.nan for c in chunks]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, 11), lift, color="tab:blue")
    ax.axhline(1, color="red", ls="--", label="Random (lift = 1)")
    for i, v in enumerate(lift):
        ax.text(i + 1, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set(title=f"Lift by score decile on Test - {winner}", xlabel="Decile (1 = highest scores)",
           ylabel="Repeat rate / overall rate")
    ax.set_xticks(range(1, 11))
    ax.legend()
    savefig(fig, dirs, "ml", "08_lift_by_decile_test.png")
 
 
def _calib_points(y, p, bins=8):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["b"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"))
    return g["p"].values, g["y"].values
 
 
def ml_calibration(winner, art, dirs):
    m, y_o = art["mask"], art["y_dev"][art["mask"]]
    fig, ax = plt.subplots(figsize=(7, 7))
    px, py = _calib_points(y_o, art["oof_cal"][winner][m])
    ax.plot(px * 100, py * 100, "o-", label="Out-of-fold (Dev)")
    tx, ty = _calib_points(art["y_te"], art["p_te"])
    ax.plot(tx * 100, ty * 100, "s-", label="Test")
    hi = max(px.max(), py.max(), tx.max(), ty.max()) * 100 * 1.1
    ax.plot([0, hi], [0, hi], "r--", label="Perfect calibration")
    ax.set(title=f"Calibration (reliability) - {winner}", xlabel="Predicted repeat probability (%)",
           ylabel="Observed repeat rate (%)")
    ax.legend()
    savefig(fig, dirs, "ml", "09_calibration_curve.png")
 
 
def ml_score_distribution(winner, art, dirs):
    y, p = art["y_te"], art["p_te"]
    hi = np.quantile(p, 0.995)
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(p.min(), hi, 40)
    ax.hist(p[y == 0], bins=bins, density=True, alpha=0.55, label="No repeat")
    ax.hist(p[y == 1], bins=bins, density=True, alpha=0.55, label="Repeat")
    ax.axvline(art["thr"], color="black", ls="--", label=f"Threshold {art['thr']:.4f}")
    ax.set(title=f"Predicted-probability distribution by true class (Test) - {winner}",
           xlabel="Predicted repeat probability", ylabel="Density")
    ax.legend()
    savefig(fig, dirs, "ml", "10_score_distribution_test.png")
 
 
def ml_threshold(winner, art, dirs):
    m = art["mask"]
    y, p = art["y_dev"][m], art["oof_cal"][winner][m]
    order = np.argsort(-p)
    cum = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    share, prec, rec = k / len(y), cum / k, cum / max(y.sum(), 1)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    lim = share <= 0.4
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(share[lim] * 100, prec[lim] * 100, label="Precision")
    ax.plot(share[lim] * 100, rec[lim] * 100, label="Recall")
    ax.plot(share[lim] * 100, f1[lim] * 100, label="F1")
    ax.axhline(y.mean() * 100, color="gray", ls=":", label="Base rate")
    ax.axvline(MAX_FLAG * 100, color="red", ls="--", label=f"Contact budget {MAX_FLAG:.0%}")
    op = (p >= art["thr"]).mean() * 100
    ax.axvline(op, color="black", ls="-.", label=f"Chosen threshold ({op:.1f}% flagged)")
    ax.set(title=f"Precision / Recall / F1 vs share of customers contacted - {winner} (out-of-fold)",
           xlabel="% of customers contacted (highest score first)", ylabel="%")
    ax.legend()
    savefig(fig, dirs, "ml", "11_threshold_analysis.png")
 
 
def ml_importance(art, dirs):
    top = art["imp"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set(title=f"Permutation importance on a held-out CV fold - {art['fi_name']}",
           xlabel="Mean decrease in PR-AUC")
    savefig(fig, dirs, "ml", "12_permutation_importance.png")
 
 
def ml_fold_stability(art, dirs):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, sc in art["fold_scores"].items():
        ax.plot(range(1, len(sc) + 1), sc, marker="o", label=name)
    ax.plot(range(1, len(art["fold_base"]) + 1), art["fold_base"], "k--", label="Random baseline")
    ax.set_xticks(range(1, len(art["fold_base"]) + 1))
    ax.set(title="PR-AUC per purged CV fold (shows how noisy model differences are)",
           xlabel="CV fold (later = more recent)", ylabel="PR-AUC")
    ax.legend(fontsize=8)
    savefig(fig, dirs, "ml", "13_cv_fold_stability.png")
 
 
# ----------------------------------------------------------------
# 7. WRITTEN REPORT
# ----------------------------------------------------------------
def _md_table(df, cols):
    t = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        t += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return t
 
 
def write_report(dirs, cust, retention, sizes, ml, stats, dev, test, results, winner, art,
                 trials, cutoff, n_rows, min_cohort):
    tm, ci = art["test_metrics"], art["ci"]
    big = sizes[sizes >= min_cohort].index
    m1 = retention.loc[big, 1].mean()
    cols = ["model", "cv_pr_auc_mean", "cv_pr_auc_std", "oof_pr_auc_pooled", "oof_roc_auc", "oof_accuracy",
            "oof_balanced_accuracy", "oof_precision", "oof_recall", "oof_f1", "oof_flagged_share"]
    table = _md_table(results[cols].round(4), cols)
    small = (f"\n- **Dataset is small** ({n_rows:,} rows): few positives, every metric is noisy."
             if n_rows < 1_000_000 else "")
    text = f"""# Customer Retention Report (v3)
 
## 1. Data summary
- Rows: **{n_rows:,}** | Customers: **{len(cust):,}** | Ever-repeat customers: **{cust['repeat'].sum():,} ({cust['repeat'].mean()*100:.2f}%)**
- Data ends: {(cutoff - pd.Timedelta(days=1)).date()} (found from order density)
- Average month-1 retention across big cohorts: **{m1:.2f}%**
 
## 2. Target definition (fixed)
- Orders placed within the first **{SESSION_HOURS}h** are treated as ONE purchase (split checkouts).
- Label = another order in the following **{FOLLOWUP_DAYS} days**; features are known at the end of the first session.
- With the old definition there were {stats['old_positives']:,} positives; **{stats['removed_share']*100:.0f}%** of them were same-session orders and are no longer counted -> {stats['new_positives']:,} positives remain.
- Eligible customers: {len(ml):,}; positive rate {ml['target'].mean()*100:.2f}%
- Dev {len(dev):,} ({dev['target'].sum()} positives, rate {dev['target'].mean()*100:.2f}%) | Test {len(test):,} ({test['target'].sum()} positives, rate {test['target'].mean()*100:.2f}%)
 
## 3. Model selection (purged time-CV on Dev; out-of-fold metrics at the contact-budget threshold)
{table}
Threshold = best F1 among thresholds that flag at most **{MAX_FLAG:.0%}** of customers. {trials} random trials per model. The Ensemble is the average of the calibrated model scores.
 
**Selected model: {winner}** (best CV PR-AUC). Accuracy is high mainly because ~98-99% of customers do not return; use PR-AUC, lift, precision and recall.
 
## 4. Test result (winner only, used once; 95% bootstrap CI)
- PR-AUC **{tm['pr_auc']:.4f}** [{ci['pr_auc'][0]:.4f}, {ci['pr_auc'][1]:.4f}] vs random baseline {test['target'].mean():.4f}
- ROC-AUC **{tm['roc_auc']:.4f}** [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}]
- Accuracy {tm['accuracy']:.4f} | Balanced accuracy {tm['balanced_accuracy']:.4f} | Precision {tm['precision']:.4f} | Recall {tm['recall']:.4f} | F1 {tm['f1']:.4f} | flags {tm['flagged_share']*100:.1f}% of customers
- Top 10% scored customers: precision {tm['precision_top10']:.3f}, captures {tm['recall_top10']*100:.1f}% of repeaters, lift **{tm['lift_top10']:.2f}x** [{ci['lift_top10'][0]:.2f}, {ci['lift_top10'][1]:.2f}]
- Brier (calibrated) {tm['brier']:.5f} vs base-rate predictor {tm['brier_baseline']:.5f}
- Top features ({art['fi_name']}, held-out fold): {', '.join(art['imp'].head(5)['feature'])}
 
## 4b. Plots
- `figures/eda/` (17 files): cohorts, orders trend, orders per customer, days-to-2nd-order, repeat rate by value / freight / distance / promised days / category / state / payment / installments / month, correlation, distributions, missing values.
- `figures/ml/` (13 files): CV comparison, metrics bars + table, PR/ROC curves, confusion matrix, gains, lift, calibration, score distribution, threshold analysis, importance, fold stability.
 
## 5. Honest limitations
- Repeat purchasing is rare, so precision/F1 stay low by construction; a model cannot create signal that the first-order data does not contain.{small}
- Test positives are few: read the confidence intervals, not point estimates.
- Dev and Test base rates can differ (see eda/10); the threshold may need re-tuning on fresh data.
- Final model is refit on all Dev, while calibration/threshold come from smaller fold models: a small mismatch.
- 'No return yet' is not permanent churn.
"""
    (dirs["reports"] / "retention_report.md").write_text(text, encoding="utf-8")
    log.info("Report saved: %s", dirs["reports"] / "retention_report.md")
 
 
# ----------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------
def main():
    global SESSION_HOURS, FOLLOWUP_DAYS, MAX_FLAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="4 trials per model")
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--session-hours", type=float, default=SESSION_HOURS)
    ap.add_argument("--followup-days", type=int, default=FOLLOWUP_DAYS)
    ap.add_argument("--max-flag", type=float, default=MAX_FLAG, help="max share of customers to flag")
    ap.add_argument("--min-date", default="2017-01-01", help="ignore customers before this date")
    ap.add_argument("--min-cohort", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    SESSION_HOURS, FOLLOWUP_DAYS, MAX_FLAG = args.session_hours, args.followup_days, args.max_flag
    trials = 4 if args.quick else args.trials
 
    dirs = make_dirs(Path(args.out) if args.out else OUTPUT_DIR)
    t_all = time.time()
 
    raw = load_raw(find_dataset(args.data))
    n_rows = len(raw)
    orders = build_orders(raw)
    del raw
    cutoff = find_cutoff(orders)
    orders = orders[orders["purchase_ts"] < cutoff].reset_index(drop=True)
    last_full_p = last_full_period(cutoff)
 
    retention, sizes, cust = cohort_analysis(orders, dirs, last_full_p)
    ml, stats = build_ml_dataset(orders, cutoff, args.min_date)
    dev, test, t_test = split_dev_test(ml, args.test_days)
    if min(dev["target"].nunique(), test["target"].nunique()) < 2:
        raise SystemExit("A split has a single target class; ML is not meaningful for this data.")
 
    # ---- EDA plots ----
    eda_cohort_heatmap(retention, sizes, dirs, args.min_cohort)
    eda_orders_trend(orders, cust, dirs, last_full_p)
    eda_orders_per_customer(cust, dirs)
    eda_days_to_second_order(orders, dirs)
    eda_rate_by_bins(ml, "order_value", "Repeat rate by first-order value", "05_repeat_by_order_value.png", dirs)
    eda_rate_by_bins(ml, "freight_ratio", "Repeat rate by freight share of order value", "06_repeat_by_freight_ratio.png", dirs)
    eda_rate_by_bins(ml, "distance_km", "Repeat rate by seller-customer distance (km)", "07_repeat_by_distance.png", dirs)
    eda_rate_by_bins(ml, "promised_days", "Repeat rate by promised delivery days", "08_repeat_by_promised_days.png", dirs)
    eda_rate_by_cat(ml, "category", "Repeat rate by product category (top 15 by size)", "09a_repeat_by_category.png", dirs)
    eda_rate_by_cat(ml, "customer_state", "Repeat rate by customer state (top 15 by size)", "09b_repeat_by_state.png", dirs)
    eda_rate_by_cat(ml, "payment_type", "Repeat rate by payment type", "09c_repeat_by_payment_type.png", dirs, min_n=100)
    eda_rate_by_cat(ml.assign(inst=ml["installments"].clip(upper=10).round().astype("Int64").astype(str)),
                    "inst", "Repeat rate by installments (10 = 10+)", "09d_repeat_by_installments.png", dirs, min_n=100)
    eda_monthly_rate(ml, t_test, dirs)
    eda_correlation(ml, dirs)
    eda_distributions(ml, dirs)
    eda_missing(ml, dirs)
    log.info("EDA plots saved (%.0fs so far)", time.time() - t_all)
 
    # ---- ML ----
    results, winner, art = train_models(dev, test, dirs, trials)
    ml_cv_comparison(results, winner, art, dirs)
    ml_metrics_bars(results, winner, art, dirs)
    ml_metrics_table(results, winner, art, dirs)
    ml_pr_curves(results, winner, art, dirs)
    ml_roc_curves(results, winner, art, dirs)
    ml_confusion(winner, art, dirs)
    ml_gains(winner, art, dirs)
    ml_lift(winner, art, dirs)
    ml_calibration(winner, art, dirs)
    ml_score_distribution(winner, art, dirs)
    ml_threshold(winner, art, dirs)
    ml_importance(art, dirs)
    ml_fold_stability(art, dirs)
    write_report(dirs, cust, retention, sizes, ml, stats, dev, test, results, winner, art,
                 trials, cutoff, n_rows, args.min_cohort)
    log.info("DONE in %.1f min. Outputs: %s", (time.time() - t_all) / 60, dirs["reports"].parent)
 
 
if __name__ == "__main__":
    main()
 





