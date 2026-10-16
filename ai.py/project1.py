
"""
CUSTOMER COHORT RETENTION ANALYSIS
Professional Olist E-commerce Analytics Pipeline

Input:
    olist_all_files_merged.csv

Outputs:
    results/  - analytical reports, CSV tables and summaries
    plots/    - clean, separately saved plots
    models/   - best trained machine-learning model

Machine Learning:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Extra Trees
    5. HistGradientBoosting

Each model is configured for 100 randomized hyperparameter trials.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance
import joblib


# =========================================================
# 1. SETTINGS
# =========================================================

SEED = 42
PREDICTION_HORIZON_DAYS = 90
TRIALS_PER_MODEL = 100

sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="deep"
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 160,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.autolayout": True,
})

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

SCRIPT_DIR = Path("Customer.py/olist_all_files_merged.csv").resolve().parent

if SCRIPT_DIR.name.lower() == "src":
    BASE = SCRIPT_DIR.parent
else:
    BASE = SCRIPT_DIR

DATA_CANDIDATES = [
    BASE / "data" / "olist_all_files_merged.csv",
    BASE / "olist_all_files_merged.csv",
    BASE / "data" / "olist_6_files_merged.csv",
    BASE / "olist_6_files_merged.csv",
    SCRIPT_DIR / "olist_all_files_merged.csv",
]

PLOTS = BASE / "plots"
RESULTS = BASE / "results"
MODELS = BASE / "models"

for folder in (PLOTS, RESULTS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("cohort_analysis")


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def save_csv(frame: pd.DataFrame, name: str) -> None:
    """Save a dataframe to the results directory."""
    frame.to_csv(
        RESULTS / name,
        index=bool(frame.index.name)
    )


def save_fig(fig, name: str) -> None:
    """Save each plot as a separate, high-quality image."""
    fig.savefig(
        PLOTS / name,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)


def locate_dataset() -> Path:
    """Find the merged CSV without requiring a manually edited path."""

    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(
        f"  - {path}" for path in DATA_CANDIDATES
    )

    raise FileNotFoundError(
        "Merged CSV could not be found.\n"
        "Place olist_all_files_merged.csv in the project "
        "root or inside the data folder.\n\n"
        f"Paths checked:\n{searched}"
    )


# =========================================================
# 5. DATA PROFILE AND QUALITY AUDIT
# =========================================================

def profile_and_quality(df: pd.DataFrame) -> None:

    log.info("Creating data profile and quality audit")

    profile = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dtype) for dtype in df.dtypes],
        "non_null": df.notna().sum().values,
        "missing_count": df.isna().sum().values,
        "missing_pct": (
            df.isna().mean().values * 100
        ).round(3),
        "n_unique": [
            df[col].nunique(dropna=True)
            for col in df.columns
        ],
    })

    profile["constant_or_single_value"] = (
        profile["n_unique"] <= 1
    )

    profile["identifier_like"] = profile.apply(
        lambda row: (
            "id" in row["column"].lower()
            and row["n_unique"] > 0
        ),
        axis=1
    )

    profile.to_csv(
        RESULTS / "dataset_profile.csv",
        index=False
    )

    # Numerical summary
    numeric = df.select_dtypes(include=np.number)

    if not numeric.empty:

        numeric.describe(
            percentiles=[
                0.01, 0.05, 0.25, 0.50,
                0.75, 0.95, 0.99
            ]
        ).T.to_csv(
            RESULTS / "numerical_summary.csv"
        )

        skewness = numeric.skew(
            numeric_only=True
        ).rename("skewness")

        kurtosis = numeric.kurt(
            numeric_only=True
        ).rename("kurtosis")

        pd.concat(
            [skewness, kurtosis],
            axis=1
        ).to_csv(
            RESULTS / "numeric_shape_summary.csv"
        )

    # Categorical summary
    categorical_rows = []

    for col in df.select_dtypes(
        include=["object", "category", "string"]
    ).columns:

        top_values = (
            df[col]
            .value_counts(dropna=False)
            .head(20)
        )

        for value, count in top_values.items():

            categorical_rows.append({
                "column": col,
                "value": str(value),
                "count": int(count),
                "pct_of_rows": round(
                    count / max(len(df), 1) * 100,
                    3
                ),
            })

    pd.DataFrame(
        categorical_rows
    ).to_csv(
        RESULTS / "categorical_top_values.csv",
        index=False
    )

    # Overall data quality
    audit = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "exact_duplicate_rows": int(
            df.duplicated().sum()
        ),
        "missing_cells": int(
            df.isna().sum().sum()
        ),
        "missing_cell_pct": round(
            float(df.isna().mean().mean() * 100),
            4
        ),
        "memory_mb": round(
            float(
                df.memory_usage(deep=True).sum()
                / 1024**2
            ),
            2
        ),
    }

    (
        RESULTS / "data_quality_summary.json"
    ).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8"
    )

    # Missingness plot
    missing = (
        df.isna().mean() * 100
    ).sort_values(ascending=False)

    missing = missing[missing > 0].head(20)
    missing = missing.sort_values()

    if not missing.empty:

        fig, ax = plt.subplots(
            figsize=(9, max(4, 0.28 * len(missing)))
        )

        missing.plot(
            kind="barh",
            ax=ax
        )

        ax.set(
            title="Top 20 Columns by Missingness",
            xlabel="Missing Values (%)",
            ylabel=""
        )

        save_fig(fig, "01_missingness.png")

    log.info(
        "Profile created: %s rows x %s columns",
        df.shape[0],
        df.shape[1]
    )


# =========================================================
# 6. PREPARE ORDER-LEVEL DATA
# =========================================================

def prepare_orders(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    required = {
        "order_id",
        "customer_unique_id",
        "order_purchase_timestamp",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Required columns missing: "
            f"{sorted(missing)}"
        )

    work = df.copy()

    work["order_purchase_timestamp"] = pd.to_datetime(
        work["order_purchase_timestamp"],
        errors="coerce"
    )

    if "order_status" in work.columns:

        work["order_status"] = (
            work["order_status"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    # One row per order
    order_agg = {
        "customer_unique_id": (
            "customer_unique_id", "first"
        ),
        "purchase_date": (
            "order_purchase_timestamp", "min"
        ),
    }

    if "order_status" in work.columns:

        order_agg["order_status"] = (
            "order_status", "first"
        )

    if "payment_total_value" in work.columns:

        # Payment totals may repeat across item-level rows.
        order_agg["payment_total_value"] = (
            "payment_total_value", "median"
        )

    if "review_score_mean" in work.columns:

        order_agg["review_score_mean"] = (
            "review_score_mean", "mean"
        )

    orders = (
        work.groupby(
            "order_id",
            sort=False,
            dropna=False
        )
        .agg(**order_agg)
        .reset_index()
    )

    # Revenue from item price + freight
    for col in ["price", "freight_value"]:

        if col not in work.columns:
            work[col] = 0.0

        work[col] = pd.to_numeric(
            work[col],
            errors="coerce"
        )

    work["item_revenue"] = (
        work["price"].fillna(0)
        + work["freight_value"].fillna(0)
    )

    order_revenue = (
        work.groupby(
            "order_id",
            sort=False
        )["item_revenue"]
        .sum(min_count=1)
    )

    orders = orders.merge(
        order_revenue.rename("order_revenue"),
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # Delivered orders are the default completed purchases.
    if "order_status" in orders.columns:

        completed = orders["order_status"].eq(
            "delivered"
        )

    else:

        completed = pd.Series(
            True,
            index=orders.index
        )

    orders["is_completed_purchase"] = (
        completed
        & orders["purchase_date"].notna()
        & orders["customer_unique_id"].notna()
    )

    orders["purchase_month"] = (
        orders["purchase_date"].dt.to_period("M")
    )

    # Audit item-level and order-level granularity
    item_counts = work.groupby(
        "order_id",
        sort=False
    ).size()

    if "customer_id" in work.columns:

        mapping = work.groupby(
            "customer_id"
        )["customer_unique_id"].nunique()

    else:

        mapping = pd.Series(dtype=int)

    merge_audit = pd.DataFrame([
        {
            "check": "input_rows_item_level",
            "value": len(work),
            "notes": "Original merged CSV rows",
        },
        {
            "check": "distinct_orders",
            "value": orders["order_id"].nunique(),
            "notes": "Unique order IDs",
        },
        {
            "check": "orders_with_multiple_item_rows",
            "value": int((item_counts > 1).sum()),
            "notes": "Multiple items are not necessarily duplicates",
        },
        {
            "check": "max_item_rows_per_order",
            "value": (
                int(item_counts.max())
                if len(item_counts) else 0
            ),
            "notes": "Order granularity check",
        },
        {
            "check": "customer_id_mapping_max_unique_customer_ids",
            "value": (
                int(mapping.max())
                if len(mapping) else 0
            ),
            "notes": "Expected maximum is normally 1",
        },
        {
            "check": "completed_orders",
            "value": int(
                orders["is_completed_purchase"].sum()
            ),
            "notes": "Delivered orders by default",
        },
    ])

    merge_audit.to_csv(
        RESULTS / "order_granularity_audit.csv",
        index=False
    )

    # Monthly order trend
    monthly = (
        orders.loc[orders["is_completed_purchase"]]
        .groupby("purchase_month")
        .agg(
            orders=("order_id", "nunique"),
            customers=("customer_unique_id", "nunique"),
            revenue=("order_revenue", "sum"),
        )
        .reset_index()
    )

    monthly["purchase_month"] = (
        monthly["purchase_month"].astype(str)
    )

    monthly.to_csv(
        RESULTS / "monthly_order_trend.csv",
        index=False
    )

    if not monthly.empty:

        fig, ax = plt.subplots(figsize=(11, 4.5))

        ax.plot(
            monthly["purchase_month"],
            monthly["orders"],
            marker="o",
            linewidth=1.8
        )

        ax.set(
            title="Delivered Orders by Purchase Month",
            xlabel="Purchase Month",
            ylabel="Orders"
        )

        ax.tick_params(
            axis="x",
            rotation=55
        )

        save_fig(fig, "02_monthly_orders.png")

    return work, orders


# =========================================================
# 7. CUSTOMER-LEVEL FEATURE ENGINEERING
# =========================================================

def customer_features(
    work: pd.DataFrame,
    orders: pd.DataFrame
) -> pd.DataFrame:

    valid_orders = orders.loc[
        orders["is_completed_purchase"]
    ].copy()

    valid_orders = valid_orders.dropna(
        subset=[
            "customer_unique_id",
            "purchase_date"
        ]
    )

    if valid_orders.empty:

        raise ValueError(
            "No valid completed orders with customer "
            "and purchase date were found."
        )

    customer = valid_orders.groupby(
        "customer_unique_id"
    ).agg(
        first_purchase=("purchase_date", "min"),
        last_purchase=("purchase_date", "max"),
        total_orders=("order_id", "nunique"),
        total_spending=("order_revenue", "sum"),
        average_order_value=("order_revenue", "mean"),
        total_order_revenue_median=(
            "order_revenue", "median"
        ),
    )

    customer["observed_tenure_days"] = (
        customer["last_purchase"]
        - customer["first_purchase"]
    ).dt.days

    customer["repeat_customer"] = (
        customer["total_orders"] >= 2
    ).astype(int)

    reference_date = (
        valid_orders["purchase_date"].max().normalize()
        + pd.Timedelta(days=1)
    )

    customer["recency_days"] = (
        reference_date
        - customer["last_purchase"].dt.normalize()
    ).dt.days

    customer["cohort_month"] = (
        customer["first_purchase"]
        .dt.to_period("M")
        .astype(str)
    )

    customer["frequency"] = customer["total_orders"]

    # Number of different product categories purchased
    if "product_category_name_english" in work.columns:

        completed_ids = set(
            valid_orders["order_id"]
        )

        category_rows = work.loc[
            work["order_id"].isin(completed_ids),
            [
                "order_id",
                "customer_unique_id",
                "product_category_name_english",
            ]
        ].dropna()

        category_rows = category_rows.drop_duplicates()

        diversity = (
            category_rows.groupby("customer_unique_id")[
                "product_category_name_english"
            ].nunique()
        )

        customer["category_diversity"] = (
            diversity.reindex(customer.index)
            .fillna(0)
            .astype(int)
        )

    customer.reset_index().to_csv(
        RESULTS / "customer_level_features.csv",
        index=False
    )

    return customer


# =========================================================
# 8. COHORT RETENTION ANALYSIS
# =========================================================

def cohort_analysis(
    customer: pd.DataFrame,
    orders: pd.DataFrame
) -> None:

    log.info("Calculating cohorts and retention")

    valid = orders.loc[
        orders["is_completed_purchase"]
    ].dropna(
        subset=[
            "customer_unique_id",
            "purchase_date"
        ]
    ).copy()

    valid["order_month"] = (
        valid["purchase_date"].dt.to_period("M")
    )

    first = (
        valid.groupby("customer_unique_id")["order_month"]
        .min()
        .rename("cohort_month")
    )

    valid = valid.join(
        first,
        on="customer_unique_id"
    )

    valid["cohort_index"] = (
        (
            valid["order_month"].dt.year
            - valid["cohort_month"].dt.year
        ) * 12
        + (
            valid["order_month"].dt.month
            - valid["cohort_month"].dt.month
        )
        + 1
    )

    active = valid[
        [
            "customer_unique_id",
            "cohort_month",
            "cohort_index",
        ]
    ].drop_duplicates()

    counts = (
        active.groupby(
            ["cohort_month", "cohort_index"]
        )["customer_unique_id"]
        .nunique()
    )

    matrix = counts.unstack(
        "cohort_index"
    ).sort_index()

    if 1 in matrix.columns:

        cohort_sizes = matrix[1].replace(
            0, np.nan
        )

        retention = (
            matrix.div(cohort_sizes, axis=0) * 100
        )

    else:

        retention = matrix.copy() * np.nan

    matrix.index = matrix.index.astype(str)
    retention.index = retention.index.astype(str)

    matrix.to_csv(
        RESULTS / "cohort_customer_counts.csv"
    )

    retention.to_csv(
        RESULTS / "cohort_retention_matrix_pct.csv"
    )

    # Cohort retention heatmap
    if not retention.empty:

        fig_width = max(
            9,
            min(16, retention.shape[1] * 0.65 + 4)
        )

        fig_height = max(
            5,
            min(13, retention.shape[0] * 0.32 + 2)
        )

        fig, ax = plt.subplots(
            figsize=(fig_width, fig_height)
        )

        sns.heatmap(
            retention,
            mask=retention.isna(),
            cmap="YlGnBu",
            annot=True,
            fmt=".0f",
            linewidths=0.35,
            linecolor="white",
            cbar_kws={
                "label": "Retention (%)"
            },
            ax=ax,
            annot_kws={"size": 7},
        )

        ax.set(
            title="Monthly Customer Cohort Retention",
            xlabel=(
                "Months Since First Purchase "
                "(Month 1 = Acquisition)"
            ),
            ylabel="Acquisition Cohort",
        )

        save_fig(
            fig,
            "03_cohort_retention_heatmap.png"
        )

    # Overall repeat-purchase rate
    repeat_rate = (
        customer["repeat_customer"].mean() * 100
        if len(customer) else np.nan
    )

    pd.DataFrame([{
        "metric": "customers_with_2plus_delivered_orders_pct",
        "value": (
            round(float(repeat_rate), 3)
            if pd.notna(repeat_rate) else np.nan
        ),
        "definition": (
            "Customers with at least two distinct delivered "
            "orders divided by customers with at least one "
            "delivered order"
        ),
    }]).to_csv(
        RESULTS / "repeat_purchase_rate.csv",
        index=False
    )

    cohort_sizes = (
        customer.groupby("cohort_month")
        .size()
        .rename("cohort_customers")
    )

    cohort_sizes.to_csv(
        RESULTS / "cohort_sizes.csv",
        header=True
    )


# =========================================================
# 9. RFM CUSTOMER SEGMENTATION
# =========================================================

def rfm_analysis(customer: pd.DataFrame) -> None:

    rfm = customer[
        [
            "recency_days",
            "frequency",
            "total_spending",
        ]
    ].copy()

    # Rank-based scoring handles skewed data more robustly.
    rfm["R_score"] = pd.qcut(
        rfm["recency_days"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    rfm["F_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["M_score"] = pd.qcut(
        rfm["total_spending"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["RFM_score"] = (
        rfm[["R_score", "F_score", "M_score"]]
        .astype(str)
        .agg("".join, axis=1)
    )

    def segment(row):

        if (
            row.R_score >= 4
            and row.F_score >= 4
            and row.M_score >= 4
        ):
            return "Champions"

        if (
            row.F_score >= 4
            and row.R_score >= 3
        ):
            return "Loyal / Frequent"

        if (
            row.R_score >= 4
            and row.F_score <= 2
        ):
            return "Recent / New"

        if (
            row.R_score <= 2
            and row.F_score >= 3
        ):
            return "At Risk - Previously Active"

        if (
            row.R_score <= 2
            and row.F_score <= 2
        ):
            return "Hibernating / Low Activity"

        return "Needs Nurturing"

    rfm["segment"] = rfm.apply(
        segment,
        axis=1
    )

    rfm.to_csv(
        RESULTS / "rfm_customer_segments.csv",
        index=True,
        index_label="customer_unique_id"
    )

    summary = rfm.groupby(
        "segment"
    ).agg(
        customers=("frequency", "size"),
        median_recency_days=("recency_days", "median"),
        mean_orders=("frequency", "mean"),
        median_spending=("total_spending", "median"),
        total_spending=("total_spending", "sum"),
    ).sort_values(
        "customers",
        ascending=False
    )

    summary.to_csv(
        RESULTS / "rfm_segment_summary.csv"
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.8)
    )

    summary["customers"].plot(
        kind="bar",
        ax=ax
    )

    ax.set(
        title="Customer Count by RFM Segment",
        xlabel="",
        ylabel="Customers"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    save_fig(
        fig,
        "04_rfm_segments.png"
    )


# =========================================================
# 10. STATISTICAL COMPARISON
# =========================================================

def statistical_comparisons(
    customer: pd.DataFrame
) -> None:

    rows = []

    repeat = customer.loc[
        customer["repeat_customer"].eq(1),
        "total_spending"
    ].dropna()

    one_time = customer.loc[
        customer["repeat_customer"].eq(0),
        "total_spending"
    ].dropna()

    if len(repeat) >= 2 and len(one_time) >= 2:

        statistic, p_value = stats.mannwhitneyu(
            repeat,
            one_time,
            alternative="two-sided"
        )

        effect_size = (
            1
            - (2 * statistic)
            / (len(repeat) * len(one_time))
        )

        rows.append({
            "comparison": (
                "Total spending: repeat vs one-time customers"
            ),
            "test": "Mann-Whitney U",
            "group_1_n": len(repeat),
            "group_1_median": repeat.median(),
            "group_2_n": len(one_time),
            "group_2_median": one_time.median(),
            "statistic": statistic,
            "p_value": p_value,
            "effect_size_rank_biserial": effect_size,
            "note": (
                "Association only. Repeat status and spending "
                "are both derived from observed customer history."
            ),
        })

    pd.DataFrame(rows).to_csv(
        RESULTS / "statistical_tests.csv",
        index=False
    )


# =========================================================
# 11. CUSTOMER EDA DASHBOARD
# =========================================================

def eda_plots(
    customer: pd.DataFrame,
    orders: pd.DataFrame
) -> None:

    # Four clearly separated plots in one dashboard.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 8)
    )

    order_cap = customer[
        "total_orders"
    ].quantile(0.99)

    spending_cap = customer[
        "total_spending"
    ].quantile(0.99)

    axes[0, 0].hist(
        customer["total_orders"].clip(
            upper=order_cap
        ),
        bins=30
    )

    axes[0, 0].set(
        title="Orders per Customer",
        xlabel="Distinct Orders (display capped at 99th percentile)",
        ylabel="Customers"
    )

    axes[0, 1].hist(
        customer["total_spending"].clip(
            upper=spending_cap
        ),
        bins=35
    )

    axes[0, 1].set(
        title="Customer Spending Distribution",
        xlabel="Total Item Revenue + Freight",
        ylabel="Customers"
    )

    one_time_spending = customer.loc[
        customer["repeat_customer"].eq(0),
        "total_spending"
    ].clip(upper=spending_cap)

    repeat_spending = customer.loc[
        customer["repeat_customer"].eq(1),
        "total_spending"
    ].clip(upper=spending_cap)

    axes[1, 0].boxplot(
        [
            one_time_spending,
            repeat_spending
        ],
        labels=["One-time", "Repeat"],
        showfliers=False
    )

    axes[1, 0].set(
        title="Spending by Repeat-Purchase Status",
        ylabel="Total Spending (display clipped)"
    )

    axes[1, 1].scatter(
        customer["total_orders"].clip(upper=10),
        customer["total_spending"].clip(
            upper=spending_cap
        ),
        alpha=0.25,
        s=10
    )

    axes[1, 1].set(
        title="Order Frequency vs Spending",
        xlabel="Distinct Orders (display capped at 10)",
        ylabel="Total Spending (display clipped)"
    )

    save_fig(
        fig,
        "05_customer_eda_dashboard.png"
    )

    # Spearman correlation
    numeric = customer.select_dtypes(
        include=np.number
    )

    correlation = numeric.corr(
        method="spearman"
    )

    correlation.to_csv(
        RESULTS / "customer_feature_spearman_correlation.csv"
    )

    if correlation.shape[1] >= 2:

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        sns.heatmap(
            correlation,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.3,
            ax=ax,
            cbar_kws={
                "label": "Spearman Correlation"
            }
        )

        ax.set_title(
            "Customer-Level Feature Correlations"
        )

        save_fig(
            fig,
            "06_customer_correlation.png"
        )


# =========================================================
# 12. MACHINE LEARNING: FIVE MODELS, 100 TRIALS EACH
# =========================================================

def predictive_model(
    customer: pd.DataFrame,
    orders: pd.DataFrame
) -> None:

    """
    Predict whether a customer places at least one delivered
    order during the 90 days after a historical cutoff.

    Five models are configured for 100 randomized trials each.
    Hyperparameters are selected on training-only validation data.
    The test set is not used for hyperparameter tuning.
    """

    log.info(
        "Preparing 90-day repeat-purchase prediction: "
        "5 models x 100 trials"
    )

    completed = orders.loc[
        orders["is_completed_purchase"]
    ].dropna(
        subset=[
            "customer_unique_id",
            "purchase_date"
        ]
    ).copy()

    max_date = (
        completed["purchase_date"].max().normalize()
    )

    cutoff = (
        max_date
        - pd.Timedelta(
            days=PREDICTION_HORIZON_DAYS
        )
    )

    horizon_end = (
        cutoff
        + pd.Timedelta(
            days=PREDICTION_HORIZON_DAYS
        )
    )

    # Historical orders only
    hist = completed.loc[
        completed["purchase_date"] < cutoff
    ]

    # Future orders used to create the target
    future = completed.loc[
        (completed["purchase_date"] >= cutoff)
        & (completed["purchase_date"] < horizon_end)
    ]

    if (
        hist["customer_unique_id"].nunique() < 300
        or len(future) < 30
    ):

        log.warning(
            "Skipping model: insufficient historical/future "
            "data for this target window."
        )

        (
            RESULTS / "model_status.txt"
        ).write_text(
            "Model skipped: insufficient historical/future "
            "data for a reliable 90-day target.\n",
            encoding="utf-8"
        )

        return

    # Create historical customer features
    hist_features = hist.groupby(
        "customer_unique_id"
    ).agg(
        hist_orders=("order_id", "nunique"),
        hist_spending=("order_revenue", "sum"),
        hist_avg_order_value=("order_revenue", "mean"),
        hist_first_purchase=("purchase_date", "min"),
        hist_last_purchase=("purchase_date", "max"),
    )

    hist_features["hist_recency_days"] = (
        cutoff
        - hist_features["hist_last_purchase"]
    ).dt.days

    hist_features["hist_tenure_days"] = (
        hist_features["hist_last_purchase"]
        - hist_features["hist_first_purchase"]
    ).dt.days

    # Target: at least one order in the future window
    labels = (
        future.groupby("customer_unique_id")["order_id"]
        .nunique()
        .gt(0)
        .astype(int)
        .rename("repeat_within_90d")
    )

    data = hist_features.join(
        labels,
        how="left"
    )

    data["repeat_within_90d"] = (
        data["repeat_within_90d"]
        .fillna(0)
        .astype(int)
    )

    data = data.drop(
        columns=[
            "hist_first_purchase",
            "hist_last_purchase"
        ]
    )

    if (
        data["repeat_within_90d"].nunique() < 2
        or data["repeat_within_90d"].value_counts().min() < 10
    ):

        log.warning(
            "Skipping model: target class distribution "
            "is too small or contains only one class."
        )

        (
            RESULTS / "model_status.txt"
        ).write_text(
            "Model skipped: target has fewer than two "
            "usable classes or too few examples.\n",
            encoding="utf-8"
        )

        return

    X = data.drop(
        columns="repeat_within_90d"
    )

    y = data["repeat_within_90d"]

    # Holdout test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    # Cap tuning sample to reduce runtime on a laptop.
    tune_n = min(
        20_000,
        len(X_train)
    )

    if tune_n < len(X_train):

        tune_idx, _ = train_test_split(
            X_train.index,
            train_size=tune_n,
            random_state=SEED,
            stratify=y_train
        )

        X_tune = X_train.loc[tune_idx]
        y_tune = y_train.loc[tune_idx]

    else:

        X_tune = X_train
        y_tune = y_train

    # Internal validation split
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tune,
        y_tune,
        test_size=0.25,
        random_state=SEED,
        stratify=y_tune
    )

    numeric_cols = X.columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scale",
                    StandardScaler()
                ),
            ]),
            numeric_cols
        )
    ], remainder="drop")

    # -----------------------------------------------------
    # FIVE MODEL CONFIGURATIONS
    # -----------------------------------------------------

    model_spaces = {

        "LogisticRegression": (
            LogisticRegression(
                max_iter=700,
                class_weight="balanced",
                random_state=SEED
            ),
            {
                "model__C": np.logspace(
                    -3, 2, 100
                ).tolist(),
                "model__solver": [
                    "liblinear",
                    "lbfgs"
                ],
            }
        ),

        "DecisionTree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=SEED
            ),
            {
                "model__max_depth": [
                    3, 5, 8, 12, 16, 20, None
                ],
                "model__min_samples_split": [
                    2, 5, 10, 20, 40, 80
                ],
                "model__min_samples_leaf": [
                    1, 2, 5, 10, 20, 40
                ],
                "model__criterion": [
                    "gini",
                    "entropy",
                    "log_loss"
                ],
            }
        ),

        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            ),
            {
                "model__n_estimators": [
                    80, 120, 180
                ],
                "model__max_depth": [
                    None, 8, 14, 20
                ],
                "model__min_samples_leaf": [
                    2, 5, 10, 20
                ],
                "model__max_features": [
                    "sqrt", 0.8
                ],
            }
        ),

        "ExtraTrees": (
            ExtraTreesClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED
            ),
            {
                "model__n_estimators": [
                    80, 120, 180
                ],
                "model__max_depth": [
                    None, 8, 14, 20
                ],
                "model__min_samples_leaf": [
                    2, 5, 10, 20
                ],
                "model__max_features": [
                    "sqrt", 0.8
                ],
            }
        ),

        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                random_state=SEED
            ),
            {
                "model__learning_rate": [
                    0.03, 0.05, 0.08, 0.1
                ],
                "model__max_iter": [
                    80, 120, 180
                ],
                "model__max_leaf_nodes": [
                    7, 15, 31
                ],
                "model__l2_regularization": [
                    0.0, 0.1, 1.0, 5.0
                ],
                "model__min_samples_leaf": [
                    15, 25, 40, 60
                ],
            }
        ),
    }

    tuning_rows = []
    best_estimators = {}

    # -----------------------------------------------------
    # RANDOMIZED HYPERPARAMETER SEARCH
    # -----------------------------------------------------

    for model_name, (estimator, space) in model_spaces.items():

        log.info(
            "Tuning %s: %d trials",
            model_name,
            TRIALS_PER_MODEL
        )

        candidates = list(
            ParameterSampler(
                space,
                n_iter=TRIALS_PER_MODEL,
                random_state=SEED
            )
        )

        best_ap = -np.inf
        best_params = None
        completed_trials = 0

        for trial_no, params in enumerate(
            candidates,
            start=1
        ):

            trial_estimator = estimator.__class__(
                **estimator.get_params()
            )

            trial_pipe = Pipeline([
                ("preprocess", preprocessor),
                ("model", trial_estimator)
            ])

            trial_pipe.set_params(
                **params
            )

            fit_kwargs = {}

            if model_name == "HistGradientBoosting":

                fit_kwargs[
                    "model__sample_weight"
                ] = compute_sample_weight(
                    "balanced",
                    y_fit
                )

            try:

                trial_pipe.fit(
                    X_fit,
                    y_fit,
                    **fit_kwargs
                )

                val_prob = (
                    trial_pipe.predict_proba(X_val)[:, 1]
                )

                ap = average_precision_score(
                    y_val,
                    val_prob
                )

                completed_trials += 1

                if ap > best_ap:

                    best_ap = ap
                    best_params = params

            except (
                ValueError,
                TypeError
            ) as exc:

                log.warning(
                    "%s trial %d skipped: %s",
                    model_name,
                    trial_no,
                    exc
                )

            if trial_no % 25 == 0:

                log.info(
                    "%s progress: %d/%d trials",
                    model_name,
                    trial_no,
                    TRIALS_PER_MODEL
                )

        if best_params is None:

            log.warning(
                "All trials failed for %s; skipping this model.",
                model_name
            )

            continue

        # Refit best configuration on all training rows.
        final_estimator = estimator.__class__(
            **estimator.get_params()
        )

        final_pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", final_estimator)
        ])

        final_pipe.set_params(
            **best_params
        )

        final_fit_kwargs = {}

        if model_name == "HistGradientBoosting":

            final_fit_kwargs[
                "model__sample_weight"
            ] = compute_sample_weight(
                "balanced",
                y_train
            )

        final_pipe.fit(
            X_train,
            y_train,
            **final_fit_kwargs
        )

        test_pred = final_pipe.predict(
            X_test
        )

        test_prob = (
            final_pipe.predict_proba(X_test)[:, 1]
        )

        row = {
            "model": model_name,
            "trials_requested": TRIALS_PER_MODEL,
            "trials_completed": completed_trials,
            "best_validation_pr_auc": best_ap,
            "accuracy": accuracy_score(
                y_test, test_pred
            ),
            "precision": precision_score(
                y_test,
                test_pred,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                test_pred,
                zero_division=0
            ),
            "f1": f1_score(
                y_test,
                test_pred,
                zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test,
                test_prob
            ),
            "pr_auc_average_precision": average_precision_score(
                y_test,
                test_prob
            ),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "tuning_rows": len(X_tune),
            "positive_rate_test": float(
                y_test.mean()
            ),
            "cutoff_date": str(cutoff.date()),
            "prediction_horizon_days": (
                PREDICTION_HORIZON_DAYS
            ),
            "best_parameters": json.dumps(
                best_params,
                default=str
            ),
        }

        tuning_rows.append(row)
        best_estimators[model_name] = final_pipe

        pd.DataFrame(
            tuning_rows
        ).to_csv(
            RESULTS / "model_comparison.csv",
            index=False
        )

    # -----------------------------------------------------
    # MODEL RESULTS AND BEST MODEL
    # -----------------------------------------------------

    if not tuning_rows:

        (
            RESULTS / "model_status.txt"
        ).write_text(
            "No model completed successfully.\n",
            encoding="utf-8"
        )

        return

    comparison = pd.DataFrame(
        tuning_rows
    ).sort_values(
        [
            "pr_auc_average_precision",
            "roc_auc"
        ],
        ascending=False
    )

    comparison.to_csv(
        RESULTS / "model_comparison.csv",
        index=False
    )

    best_name = comparison.iloc[0]["model"]
    best = best_estimators[best_name]

    joblib.dump(
        best,
        MODELS / "customer_repeat_purchase_pipeline.joblib"
    )

    best_pred = best.predict(
        X_test
    )

    best_prob = (
        best.predict_proba(X_test)[:, 1]
    )

    report = classification_report(
        y_test,
        best_pred,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        RESULTS / "best_model_classification_report.csv"
    )

    status = (
        f"Target: at least one delivered order in the "
        f"90 days after {cutoff.date()}.\n"
        "Observation population: customers with at least "
        "one delivered order before cutoff.\n"
        f"Models completed: {len(tuning_rows)}.\n"
        f"Trials requested per model: {TRIALS_PER_MODEL}.\n"
        f"Best model selected by holdout PR-AUC: {best_name}.\n"
        "Hyperparameters were tuned on training-only "
        "validation data; test data was not used for tuning.\n"
        "Tuning uses at most 20,000 training rows for runtime; "
        "final models are refit on all training rows.\n"
        "This is an exploratory single-cutoff evaluation, "
        "not a rolling-origin temporal benchmark.\n"
    )

    (
        RESULTS / "model_status.txt"
    ).write_text(
        status,
        encoding="utf-8"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        best_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4.5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "No Order",
            "Order"
        ],
        yticklabels=[
            "No Order",
            "Order"
        ],
        ax=ax
    )

    ax.set(
        title=f"Confusion Matrix — {best_name}",
        xlabel="Predicted",
        ylabel="Actual"
    )

    save_fig(
        fig,
        "07_model_confusion_matrix.png"
    )

    # Model comparison plot
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    comparison.set_index("model")[
        [
            "roc_auc",
            "pr_auc_average_precision",
            "f1"
        ]
    ].plot(
        kind="bar",
        ax=ax
    )

    ax.set(
        title="Five Tuned Models — Holdout Comparison",
        ylabel="Score",
        xlabel="Model"
    )

    ax.set_ylim(0, 1)

    ax.tick_params(
        axis="x",
        rotation=20
    )

    save_fig(
        fig,
        "08_model_comparison.png"
    )

    # Permutation feature importance
    try:

        permutation = permutation_importance(
            best,
            X_test,
            y_test,
            n_repeats=3,
            random_state=SEED,
            scoring="average_precision",
            n_jobs=-1
        )

        importance = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": (
                permutation.importances_mean
            ),
            "importance_std": (
                permutation.importances_std
            ),
        }).sort_values(
            "importance_mean",
            ascending=False
        )

        importance.to_csv(
            RESULTS / "model_permutation_importance.csv",
            index=False
        )

        top = importance.head(10).sort_values(
            "importance_mean"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4.5)
        )

        ax.barh(
            top["feature"],
            top["importance_mean"],
            xerr=top["importance_std"]
        )

        ax.set(
            title="Top Predictive Features",
            xlabel=(
                "Decrease in PR-AUC when feature is permuted"
            ),
            ylabel=""
        )

        save_fig(
            fig,
            "09_model_feature_importance.png"
        )

    except Exception as exc:

        log.warning(
            "Permutation importance skipped: %s",
            exc
        )


# =========================================================
# 13. FINAL RUN SUMMARY
# =========================================================

def write_readme_summary(
    df: pd.DataFrame,
    orders: pd.DataFrame,
    customer: pd.DataFrame
) -> None:

    completed = orders.loc[
        orders["is_completed_purchase"]
    ]

    summary = [
        "CUSTOMER COHORT RETENTION ANALYSIS — RUN SUMMARY",
        "=" * 52,
        f"Input rows: {len(df):,}",
        f"Input columns: {df.shape[1]:,}",
        f"Distinct orders: {orders['order_id'].nunique():,}",
        f"Delivered/completed orders used: {len(completed):,}",
        f"Customers with >=1 delivered order: {customer.shape[0]:,}",
        (
            "Customers with 2+ delivered orders: "
            f"{int(customer['repeat_customer'].sum()):,}"
        ),
        (
            "Repeat-purchase rate: "
            f"{customer['repeat_customer'].mean() * 100:.2f}%"
        ),
        (
            "Purchase period: "
            f"{orders['purchase_date'].min()} to "
            f"{orders['purchase_date'].max()}"
        ),
        "",
        "Definitions:",
        (
            "- Cohort = month of first observed delivered "
            "order in this dataset."
        ),
        (
            "- Monthly retention = cohort customers with at "
            "least one delivered order in that month / "
            "original cohort size."
        ),
        (
            "- Repeat purchase = at least two distinct "
            "delivered orders over observed history; "
            "this is not monthly retention."
        ),
        (
            "- Spending = item price + freight within an "
            "order, aggregated by customer."
        ),
        (
            "- RFM segments are heuristic rank-based groups, "
            "not ground-truth labels."
        ),
        (
            "- Predictions are exploratory and require "
            "stronger temporal validation before operational use."
        ),
        "",
        "See results/ and plots/ for detailed outputs.",
    ]

    (
        RESULTS / "run_summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8"
    )


# =========================================================
# 14. MAIN EXECUTION
# =========================================================

def main() -> None:

    start = time.perf_counter()

    dataset_path = locate_dataset()

    log.info(
        "Loading dataset: %s",
        dataset_path
    )

    df = pd.read_csv(
        dataset_path,
        low_memory=False
    )

    log.info(
        "Loaded %s rows x %s columns",
        *df.shape
    )

    profile_and_quality(df)

    work, orders = prepare_orders(df)

    customer = customer_features(
        work,
        orders
    )

    cohort_analysis(
        customer,
        orders
    )

    rfm_analysis(
        customer
    )

    statistical_comparisons(
        customer
    )

    eda_plots(
        customer,
        orders
    )

    predictive_model(
        customer,
        orders
    )

    write_readme_summary(
        df,
        orders,
        customer
    )

    elapsed = time.perf_counter() - start

    log.info(
        "COMPLETE in %.1f seconds. Outputs: %s, %s, %s",
        elapsed,
        RESULTS,
        PLOTS,
        MODELS
    )


if __name__ == "__main__":
    main()
