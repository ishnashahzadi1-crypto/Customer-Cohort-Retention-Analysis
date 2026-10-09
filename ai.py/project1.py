# ============================================================
# CUSTOMER COHORT RETENTION ANALYSIS
# PART 1 — HISTORICAL RETENTION ANALYSIS
#
# Purpose:
# Track retention rates across acquisition cohorts to understand
# long-term customer engagement, repeat-purchase behavior,
# customer loyalty patterns, and retained vs non-retained behavior.
#
# Designed for:
# - OLIST merged dataset
# - VS Code / Windows
# - Professional portfolio / GitHub
# - Fast + memory-conscious processing
#
# RANDOM_STATE = 42
# ============================================================

from pathlib import Path
import warnings
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import (
    ttest_ind,
    mannwhitneyu,
    chi2_contingency,
    levene,
)

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path("Customer.py/olist_customer_cohort_retention_comprehensive.csv").resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"

for folder in [TABLE_DIR, PLOT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ============================================================
# 2. DATASET FINDER
# ============================================================

POSSIBLE_FILES = [
    PROJECT_ROOT / "olist_6_files_merged(1).csv",
    PROJECT_ROOT / "olist_6_files_merged.csv",
    PROJECT_ROOT / "olist_6_files_merged(2).csv",
    PROJECT_ROOT / "olist.csv",
    PROJECT_ROOT / "data" / "olist_6_files_merged(1).csv",
    PROJECT_ROOT / "data" / "olist_6_files_merged.csv",
    PROJECT_ROOT / "data" / "olist.csv",
]


def find_dataset():
    """Find the first available OLIST dataset."""

    for path in POSSIBLE_FILES:
        if path.exists():
            return path

    csv_files = list(PROJECT_ROOT.glob("*.csv"))

    if not csv_files:
        data_dir = PROJECT_ROOT / "data"
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))

    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        "\nNo CSV dataset found.\n"
        "Place your OLIST merged CSV in the project folder or data/ folder."
    )


DATA_PATH = find_dataset()

print("=" * 80)
print("CUSTOMER COHORT RETENTION ANALYSIS — PART 1")
print("=" * 80)
print(f"Dataset: {DATA_PATH}")
print("=" * 80)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\nDATA LOADED")
print("-" * 80)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")


# ============================================================
# 4. COLUMN NORMALIZATION
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def save_csv(dataframe, filename):
    path = TABLE_DIR / filename
    dataframe.to_csv(path, index=False)
    return path


def save_plot(filename):
    path = PLOT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def detect_column(candidates, required=True):
    """
    Find a column using several possible names.
    """

    for candidate in candidates:
        candidate = candidate.lower()

        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            "\nRequired column not found.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def cramers_v(table):
    """Calculate Cramer's V."""

    if table.empty:
        return np.nan

    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()

    if n == 0:
        return np.nan

    r, k = table.shape

    if min(r, k) <= 1:
        return np.nan

    phi2 = chi2 / n

    correction = max(
        0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1)
    )

    corrected_r = r - ((r - 1) ** 2) / (n - 1)
    corrected_k = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(corrected_k - 1, corrected_r - 1)

    if denominator <= 0:
        return np.nan

    return math.sqrt(correction / denominator)


# ============================================================
# 6. COLUMN DETECTION
# ============================================================

customer_col = detect_column(
    [
        "customer_unique_id",
        "customer_id"
    ]
)

order_col = detect_column(
    [
        "order_id"
    ]
)

date_col = detect_column(
    [
        "order_purchase_timestamp",
        "purchase_timestamp",
        "order_date",
        "purchase_date"
    ]
)

price_col = detect_column(
    [
        "price",
        "order_price",
        "total_price"
    ],
    required=False
)

freight_col = detect_column(
    [
        "freight_value",
        "shipping_cost",
        "freight"
    ],
    required=False
)

order_status_col = detect_column(
    [
        "order_status"
    ],
    required=False
)

print("\nDETECTED COLUMNS")
print("-" * 80)
print(f"Customer : {customer_col}")
print(f"Order    : {order_col}")
print(f"Date     : {date_col}")
print(f"Price    : {price_col}")
print(f"Freight  : {freight_col}")
print(f"Status   : {order_status_col}")


# ============================================================
# 7. DATA QUALITY
# ============================================================

quality_rows = []

for col in df.columns:

    quality_rows.append(
        {
            "column": col,
            "dtype": str(df[col].dtype),
            "rows": len(df),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(
                df[col].isna().mean() * 100, 4
            ),
            "unique_count": int(df[col].nunique(dropna=True)),
            "unique_pct": round(
                df[col].nunique(dropna=True) / len(df) * 100,
                4
            ),
        }
    )

quality_summary = pd.DataFrame(quality_rows)

save_csv(
    quality_summary,
    "01_data_quality_summary.csv"
)


# ============================================================
# 8. REMOVE DUPLICATE EXACT ROWS
# ============================================================

initial_rows = len(df)

df = df.drop_duplicates().copy()

duplicates_removed = initial_rows - len(df)

print("\nDATA CLEANING")
print("-" * 80)
print(f"Duplicate rows removed: {duplicates_removed:,}")


# ============================================================
# 9. DATE PARSING
# ============================================================

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

invalid_dates = int(df[date_col].isna().sum())

print(f"Invalid/missing purchase dates: {invalid_dates:,}")


# ============================================================
# 10. VALID PURCHASE FILTER
# ============================================================

valid_mask = (
    df[customer_col].notna()
    & df[order_col].notna()
    & df[date_col].notna()
)

if order_status_col:

    valid_statuses = [
        "delivered"
    ]

    status_mask = (
        df[order_status_col]
        .astype(str)
        .str.lower()
        .isin(valid_statuses)
    )

    # If delivered rows exist, use them.
    # Otherwise keep all valid orders instead of returning
    # an empty dataset.
    if status_mask.sum() > 0:
        valid_mask &= status_mask

retention_df = df.loc[valid_mask].copy()

print(f"Valid purchase rows: {len(retention_df):,}")


# ============================================================
# 11. REMOVE INVALID NUMERIC VALUES
# ============================================================

if price_col:

    retention_df[price_col] = pd.to_numeric(
        retention_df[price_col],
        errors="coerce"
    )

    retention_df.loc[
        retention_df[price_col] < 0,
        price_col
    ] = np.nan

if freight_col:

    retention_df[freight_col] = pd.to_numeric(
        retention_df[freight_col],
        errors="coerce"
    )

    retention_df.loc[
        retention_df[freight_col] < 0,
        freight_col
    ] = np.nan


# ============================================================
# 12. CREATE ORDER-LEVEL DATASET
# ============================================================

order_group = [
    customer_col,
    order_col,
    date_col
]

agg_dict = {}

if price_col:
    agg_dict[price_col] = "sum"

if freight_col:
    agg_dict[freight_col] = "sum"

order_df = (
    retention_df
    .groupby(order_group, as_index=False)
    .agg(agg_dict)
)

# Rename numeric fields
if price_col:
    order_df["order_value"] = order_df[price_col]
else:
    order_df["order_value"] = np.nan

if freight_col:
    order_df["freight_value"] = order_df[freight_col]
else:
    order_df["freight_value"] = np.nan


# ============================================================
# 13. MONTHLY ORDER ACTIVITY
# ============================================================

order_df["purchase_month"] = (
    order_df[date_col]
    .dt.to_period("M")
)

order_df["purchase_month_date"] = (
    order_df["purchase_month"]
    .dt.to_timestamp()
)


# ============================================================
# 14. CUSTOMER FIRST PURCHASE
# ============================================================

first_purchase = (
    order_df
    .groupby(customer_col)[date_col]
    .min()
    .rename("first_purchase_date")
)

last_purchase = (
    order_df
    .groupby(customer_col)[date_col]
    .max()
    .rename("last_purchase_date")
)


# ============================================================
# 15. CUSTOMER-LEVEL FEATURE ENGINEERING
# ============================================================

customer_features = (
    order_df
    .groupby(customer_col)
    .agg(
        total_orders=(order_col, "nunique"),
        total_spend=("order_value", "sum"),
        average_order_value=("order_value", "mean"),
        total_freight=("freight_value", "sum"),
        active_months=("purchase_month", "nunique"),
    )
    .reset_index()
)

customer_features = customer_features.merge(
    first_purchase.reset_index(),
    on=customer_col,
    how="left"
)

customer_features = customer_features.merge(
    last_purchase.reset_index(),
    on=customer_col,
    how="left"
)

customer_features["customer_lifetime_days"] = (
    customer_features["last_purchase_date"]
    - customer_features["first_purchase_date"]
).dt.days

customer_features["repeat_customer"] = (
    customer_features["total_orders"] > 1
).astype(int)

customer_features["retained_customer"] = (
    customer_features["active_months"] > 1
).astype(int)


# ============================================================
# 16. ACQUISITION COHORT
# ============================================================

customer_features["cohort_month"] = (
    customer_features["first_purchase_date"]
    .dt.to_period("M")
)

customer_features["cohort_month_date"] = (
    customer_features["cohort_month"]
    .dt.to_timestamp()
)


# ============================================================
# 17. RETENTION DEFINITION
# ============================================================

# Historical retention definition:
#
# Retained customer = customer made purchases in more than
# one calendar month during the observed dataset period.
#
# Non-retained = customer purchased in only one calendar month.
#
# This definition is appropriate for Part 1 historical analysis.
#
# IMPORTANT:
# It does NOT mean that a customer will never return in the future.
# It only describes behavior inside the available observation window.


# ============================================================
# 18. SAVE CUSTOMER SUMMARY
# ============================================================

customer_summary_cols = [
    customer_col,
    "first_purchase_date",
    "last_purchase_date",
    "cohort_month_date",
    "total_orders",
    "total_spend",
    "average_order_value",
    "total_freight",
    "active_months",
    "customer_lifetime_days",
    "repeat_customer",
    "retained_customer",
]

customer_features[
    customer_summary_cols
].to_csv(
    TABLE_DIR / "02_customer_summary.csv",
    index=False
)

customer_features[
    customer_summary_cols
].to_csv(
    TABLE_DIR / "04_customer_features.csv",
    index=False
)


# ============================================================
# 19. MONTHLY CUSTOMER ACTIVITY
# ============================================================

monthly_customer_activity = (
    order_df[
        [
            customer_col,
            "purchase_month",
            "purchase_month_date"
        ]
    ]
    .drop_duplicates()
)

monthly_active = (
    monthly_customer_activity
    .groupby("purchase_month_date")[customer_col]
    .nunique()
    .reset_index(name="active_customers")
)

monthly_orders = (
    order_df
    .groupby("purchase_month_date")[order_col]
    .nunique()
    .reset_index(name="orders")
)

monthly_activity = monthly_active.merge(
    monthly_orders,
    on="purchase_month_date",
    how="left"
)

save_csv(
    monthly_activity,
    "03_monthly_customer_activity.csv"
)


# ============================================================
# 20. NEW VS RETURNING CUSTOMERS
# ============================================================

monthly_customer_activity = monthly_customer_activity.merge(
    customer_features[
        [
            customer_col,
            "first_purchase_date"
        ]
    ],
    on=customer_col,
    how="left"
)

monthly_customer_activity["first_purchase_month"] = (
    monthly_customer_activity["first_purchase_date"]
    .dt.to_period("M")
)

monthly_customer_activity["customer_type"] = np.where(
    monthly_customer_activity["purchase_month"]
    == monthly_customer_activity["first_purchase_month"],
    "New",
    "Returning"
)

new_returning = (
    monthly_customer_activity
    .groupby(
        [
            "purchase_month_date",
            "customer_type"
        ]
    )[customer_col]
    .nunique()
    .reset_index(name="customers")
)

save_csv(
    new_returning,
    "11_new_vs_returning_customers.csv"
)


# ============================================================
# 21. COHORT SIZE
# ============================================================

cohort_sizes = (
    customer_features
    .groupby("cohort_month_date")[customer_col]
    .nunique()
    .reset_index(name="cohort_size")
)

save_csv(
    cohort_sizes,
    "05_cohort_sizes.csv"
)


# ============================================================
# 22. CREATE RELATIVE MONTH
# ============================================================

monthly_customer_activity["activity_month"] = (
    monthly_customer_activity["purchase_month"]
)

monthly_customer_activity = monthly_customer_activity.merge(
    customer_features[
        [
            customer_col,
            "cohort_month"
        ]
    ],
    on=customer_col,
    how="left"
)

monthly_customer_activity["cohort_index"] = (
    (
        monthly_customer_activity["activity_month"]
        .dt.year
        - monthly_customer_activity["cohort_month"]
        .dt.year
    ) * 12
    +
    (
        monthly_customer_activity["activity_month"]
        .dt.month
        - monthly_customer_activity["cohort_month"]
        .dt.month
    )
)


# ============================================================
# 23. COHORT RETENTION COUNTS
# ============================================================

cohort_retention_counts = (
    monthly_customer_activity
    .groupby(
        [
            "cohort_month",
            "cohort_index"
        ]
    )[customer_col]
    .nunique()
    .reset_index(name="active_customers")
)

cohort_retention_counts = cohort_retention_counts[
    cohort_retention_counts["cohort_index"] >= 0
].copy()

cohort_retention_counts = cohort_retention_counts.merge(
    cohort_sizes.rename(
        columns={
            "cohort_month_date": "cohort_month"
        }
    ),
    on="cohort_month",
    how="left"
)

save_csv(
    cohort_retention_counts,
    "06_cohort_retention_counts.csv"
)


# ============================================================
# 24. RETENTION RATES
# ============================================================

cohort_retention_counts["retention_rate_pct"] = (
    cohort_retention_counts["active_customers"]
    /
    cohort_retention_counts["cohort_size"]
    * 100
)

save_csv(
    cohort_retention_counts,
    "07_cohort_retention_rates.csv"
)


# ============================================================
# 25. RETENTION MATRIX
# ============================================================

retention_matrix = (
    cohort_retention_counts
    .pivot(
        index="cohort_month",
        columns="cohort_index",
        values="retention_rate_pct"
    )
    .sort_index()
)

retention_matrix.columns.name = "months_since_acquisition"

retention_matrix.to_csv(
    TABLE_DIR / "08_cohort_retention_matrix.csv"
)


# ============================================================
# 26. RETENTION COUNTS MATRIX
# ============================================================

retention_count_matrix = (
    cohort_retention_counts
    .pivot(
        index="cohort_month",
        columns="cohort_index",
        values="active_customers"
    )
    .sort_index()
)

retention_count_matrix.to_csv(
    TABLE_DIR / "09_cohort_retention_count_matrix.csv"
)


# ============================================================
# 27. MONTHLY RETENTION CURVE
# ============================================================

curve_data = (
    cohort_retention_counts
    .groupby("cohort_index")
    .agg(
        average_retention_pct=(
            "retention_rate_pct",
            "mean"
        ),
        median_retention_pct=(
            "retention_rate_pct",
            "median"
        ),
        cohorts_available=(
            "cohort_month",
            "nunique"
        )
    )
    .reset_index()
)

save_csv(
    curve_data,
    "10_retention_curve_summary.csv"
)


# ============================================================
# 28. RETAINED VS NON-RETAINED
# ============================================================

retained_summary = (
    customer_features
    .groupby("retained_customer")
    .agg(
        customers=(customer_col, "nunique"),
        average_orders=("total_orders", "mean"),
        median_orders=("total_orders", "median"),
        average_spend=("total_spend", "mean"),
        median_spend=("total_spend", "median"),
        average_active_months=("active_months", "mean"),
        average_lifetime_days=(
            "customer_lifetime_days",
            "mean"
        ),
    )
    .reset_index()
)

retained_summary["customer_status"] = (
    retained_summary["retained_customer"]
    .map(
        {
            0: "Non-Retained",
            1: "Retained"
        }
    )
)

save_csv(
    retained_summary,
    "12_retained_vs_nonretained_summary.csv"
)


# ============================================================
# 29. RETAINED VS NON-RETAINED STATISTICAL TESTS
# ============================================================

statistical_results = []

numeric_features = [
    "total_orders",
    "total_spend",
    "average_order_value",
    "total_freight",
    "active_months",
    "customer_lifetime_days",
]

for feature in numeric_features:

    if feature not in customer_features.columns:
        continue

    retained_values = customer_features.loc[
        customer_features["retained_customer"] == 1,
        feature
    ].dropna()

    non_retained_values = customer_features.loc[
        customer_features["retained_customer"] == 0,
        feature
    ].dropna()

    if len(retained_values) < 2 or len(non_retained_values) < 2:
        continue

    # Welch's t-test
    t_stat, t_p = ttest_ind(
        retained_values,
        non_retained_values,
        equal_var=False
    )

    # Mann-Whitney U
    try:
        u_stat, mw_p = mannwhitneyu(
            retained_values,
            non_retained_values,
            alternative="two-sided"
        )
    except Exception:
        u_stat, mw_p = np.nan, np.nan

    statistical_results.append(
        {
            "feature": feature,
            "retained_n": len(retained_values),
            "non_retained_n": len(non_retained_values),
            "retained_mean": retained_values.mean(),
            "non_retained_mean": non_retained_values.mean(),
            "retained_median": retained_values.median(),
            "non_retained_median": non_retained_values.median(),
            "welch_t_stat": t_stat,
            "welch_t_pvalue": t_p,
            "mann_whitney_u": u_stat,
            "mann_whitney_pvalue": mw_p,
        }
    )

statistical_tests = pd.DataFrame(statistical_results)

if not statistical_tests.empty:

    # Benjamini-Hochberg FDR
    pvals = statistical_tests[
        "mann_whitney_pvalue"
    ].fillna(1).values

    order = np.argsort(pvals)
    ranked = pvals[order]

    n = len(ranked)

    adjusted = np.empty(n)

    previous = 1.0

    for i in range(n - 1, -1, -1):

        rank = i + 1

        value = ranked[i] * n / rank

        previous = min(previous, value)

        adjusted[i] = previous

    fdr_values = np.empty(n)
    fdr_values[order] = adjusted

    statistical_tests[
        "mann_whitney_fdr_pvalue"
    ] = fdr_values

    statistical_tests[
        "significant_after_fdr"
    ] = (
        statistical_tests[
            "mann_whitney_fdr_pvalue"
        ] < 0.05
    )

save_csv(
    statistical_tests,
    "13_statistical_tests.csv"
)


# ============================================================
# 30. COHORT COMPARISON
# ============================================================

cohort_comparison = (
    customer_features
    .groupby("cohort_month_date")
    .agg(
        customers=(customer_col, "nunique"),
        average_orders=("total_orders", "mean"),
        median_orders=("total_orders", "median"),
        average_spend=("total_spend", "mean"),
        median_spend=("total_spend", "median"),
        retained_rate_pct=("retained_customer", "mean"),
    )
    .reset_index()
)

cohort_comparison["retained_rate_pct"] *= 100

save_csv(
    cohort_comparison,
    "14_cohort_comparison.csv"
)


# ============================================================
# 31. OVERALL RETENTION SUMMARY
# ============================================================

total_customers = customer_features[customer_col].nunique()

retained_customers = int(
    customer_features["retained_customer"].sum()
)

non_retained_customers = (
    total_customers - retained_customers
)

retained_pct = (
    retained_customers
    / total_customers
    * 100
)

non_retained_pct = (
    non_retained_customers
    / total_customers
    * 100
)

overall_summary = pd.DataFrame(
    [
        {
            "metric": "total_customers",
            "value": total_customers
        },
        {
            "metric": "retained_customers",
            "value": retained_customers
        },
        {
            "metric": "non_retained_customers",
            "value": non_retained_customers
        },
        {
            "metric": "retained_percentage",
            "value": retained_pct
        },
        {
            "metric": "non_retained_percentage",
            "value": non_retained_pct
        },
        {
            "metric": "total_orders",
            "value": order_df[order_col].nunique()
        },
        {
            "metric": "cohorts",
            "value": cohort_sizes.shape[0]
        },
    ]
)

save_csv(
    overall_summary,
    "15_overall_retention_summary.csv"
)


# ============================================================
# 32. PLOT 01 — MONTHLY ACTIVE CUSTOMERS
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_activity["purchase_month_date"],
    monthly_activity["active_customers"],
    marker="o"
)

plt.title(
    "Monthly Active Customers",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Purchase Month")
plt.ylabel("Active Customers")

plt.xticks(rotation=45)

save_plot(
    "01_monthly_active_customers.png"
)


# ============================================================
# 33. PLOT 02 — NEW VS RETURNING
# ============================================================

new_returning_pivot = (
    new_returning
    .pivot(
        index="purchase_month_date",
        columns="customer_type",
        values="customers"
    )
    .fillna(0)
)

plt.figure(figsize=(12, 6))

new_returning_pivot.plot(
    kind="bar",
    ax=plt.gca()
)

plt.title(
    "New vs Returning Customers",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Purchase Month")
plt.ylabel("Customers")

plt.xticks(rotation=45)

save_plot(
    "02_new_vs_returning_customers.png"
)


# ============================================================
# 34. PLOT 03 — CUSTOMER ORDER DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    customer_features["total_orders"],
    bins=30,
    kde=True
)

plt.title(
    "Customer Order Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Total Orders")
plt.ylabel("Customers")

save_plot(
    "03_customer_order_distribution.png"
)


# ============================================================
# 35. PLOT 04 — CUSTOMER SPEND DISTRIBUTION
# ============================================================

if customer_features["total_spend"].notna().any():

    plot_values = customer_features[
        "total_spend"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(plot_values) > 0:

        if len(plot_values) > 20000:

            plot_values = plot_values.sample(
                20000,
                random_state=RANDOM_STATE
            )

        plt.figure(figsize=(10, 6))

        sns.histplot(
            plot_values,
            bins=40,
            kde=True
        )

        plt.title(
            "Customer Spend Distribution",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlabel("Total Spend")
        plt.ylabel("Customers")

        save_plot(
            "04_customer_spend_distribution.png"
        )


# ============================================================
# 36. PLOT 05 — COHORT SIZE
# ============================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=cohort_sizes,
    x="cohort_month_date",
    y="cohort_size"
)

plt.title(
    "Customer Acquisition Cohort Size",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Acquisition Cohort")
plt.ylabel("Customers")

plt.xticks(rotation=45)

save_plot(
    "05_cohort_size.png"
)


# ============================================================
# 37. PLOT 06 — RETENTION HEATMAP
# ============================================================

plt.figure(figsize=(14, 9))

sns.heatmap(
    retention_matrix,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    linewidths=0.5,
    cbar_kws={
        "label": "Retention %"
    }
)

plt.title(
    "Customer Cohort Retention Heatmap",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel("Months Since Acquisition")
plt.ylabel("Acquisition Cohort")

save_plot(
    "06_cohort_retention_heatmap.png"
)


# ============================================================
# 38. PLOT 07 — RETENTION CURVES
# ============================================================

plt.figure(figsize=(12, 7))

for cohort, group in cohort_retention_counts.groupby(
    "cohort_month"
):

    if len(group) >= 2:

        plt.plot(
            group["cohort_index"],
            group["retention_rate_pct"],
            marker="o",
            alpha=0.6,
            linewidth=1.5,
            label=str(cohort)
        )

plt.title(
    "Retention Curves by Acquisition Cohort",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Months Since Acquisition")
plt.ylabel("Retention Rate (%)")

# Avoid unreadable legend if there are many cohorts
if cohort_retention_counts["cohort_month"].nunique() <= 15:
    plt.legend(
        title="Cohort",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

save_plot(
    "07_retention_curves.png"
)


# ============================================================
# 39. PLOT 08 — RETAINED VS NON-RETAINED
# ============================================================

comparison_features = [
    "total_orders",
    "total_spend",
    "active_months",
    "customer_lifetime_days",
]

available_features = [
    x
    for x in comparison_features
    if x in customer_features.columns
]

if available_features:

    comparison_long = (
        customer_features[
            ["retained_customer"] + available_features
        ]
        .copy()
    )

    comparison_long["customer_status"] = (
        comparison_long["retained_customer"]
        .map(
            {
                0: "Non-Retained",
                1: "Retained"
            }
        )
    )

    # Use standardized medians to compare features
    comparison_plot = []

    for feature in available_features:

        values = comparison_long[feature]

        if values.notna().sum() == 0:
            continue

        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        temp = comparison_long[
            [
                "customer_status",
                feature
            ]
        ].dropna().copy()

        temp["standardized_value"] = (
            temp[feature] - median
        ) / iqr

        temp["feature"] = feature

        comparison_plot.append(
            temp[
                [
                    "customer_status",
                    "standardized_value",
                    "feature"
                ]
            ]
        )

    if comparison_plot:

        comparison_plot = pd.concat(
            comparison_plot,
            ignore_index=True
        )

        plt.figure(figsize=(13, 7))

        sns.boxplot(
            data=comparison_plot,
            x="feature",
            y="standardized_value",
            hue="customer_status"
        )

        plt.title(
            "Retained vs Non-Retained Customer Behavior",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlabel("Feature")
        plt.ylabel("Relative Value (Median/IQR Scaled)")

        plt.xticks(rotation=30)

        save_plot(
            "08_retained_vs_nonretained_comparison.png"
        )


# ============================================================
# 40. PLOT 09 — RETENTION RATE BY MONTH
# ============================================================

plt.figure(figsize=(11, 6))

sns.lineplot(
    data=curve_data,
    x="cohort_index",
    y="average_retention_pct",
    marker="o"
)

plt.title(
    "Average Cohort Retention Curve",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Months Since Acquisition")
plt.ylabel("Average Retention (%)")

save_plot(
    "09_average_retention_curve.png"
)


# ============================================================
# 41. PLOT 10 — RETAINED CUSTOMER BALANCE
# ============================================================

status_plot = pd.DataFrame(
    {
        "status": [
            "Retained",
            "Non-Retained"
        ],
        "customers": [
            retained_customers,
            non_retained_customers
        ]
    }
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=status_plot,
    x="status",
    y="customers"
)

plt.title(
    "Retained vs Non-Retained Customers",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Customer Status")
plt.ylabel("Customers")

for i, value in enumerate(
    status_plot["customers"]
):
    plt.text(
        i,
        value,
        f"{value:,}",
        ha="center",
        va="bottom"
    )

save_plot(
    "10_retained_customer_balance.png"
)


# ============================================================
# 42. CORRELATION ANALYSIS
# ============================================================

correlation_features = [
    "total_orders",
    "total_spend",
    "average_order_value",
    "total_freight",
    "active_months",
    "customer_lifetime_days",
    "retained_customer",
]

correlation_features = [
    col
    for col in correlation_features
    if col in customer_features.columns
]

if len(correlation_features) >= 2:

    corr = customer_features[
        correlation_features
    ].corr(
        method="spearman"
    )

    corr.to_csv(
        TABLE_DIR / "16_spearman_customer_correlation.csv"
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Customer Behavior — Spearman Correlation",
        fontsize=16,
        fontweight="bold"
    )

    save_plot(
        "11_customer_behavior_correlation.png"
    )


# ============================================================
# 43. OUTLIER SUMMARY
# ============================================================

outlier_rows = []

for feature in [
    "total_orders",
    "total_spend",
    "average_order_value",
    "total_freight",
    "active_months",
    "customer_lifetime_days",
]:

    if feature not in customer_features.columns:
        continue

    values = (
        customer_features[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(values) < 10:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (values < lower)
        |
        (values > upper)
    )

    outlier_rows.append(
        {
            "feature": feature,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(outliers.sum()),
            "outlier_pct": (
                outliers.mean() * 100
            ),
        }
    )

outlier_summary = pd.DataFrame(
    outlier_rows
)

save_csv(
    outlier_summary,
    "17_customer_outlier_summary.csv"
)


# ============================================================
# 44. SKEWNESS ANALYSIS
# ============================================================

skew_rows = []

for feature in [
    "total_orders",
    "total_spend",
    "average_order_value",
    "total_freight",
    "active_months",
    "customer_lifetime_days",
]:

    if feature not in customer_features.columns:
        continue

    values = (
        customer_features[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(values) < 10:
        continue

    skew_value = values.skew()

    skew_rows.append(
        {
            "feature": feature,
            "skewness": skew_value,
            "absolute_skewness": abs(skew_value),
            "interpretation": (
                "Highly skewed"
                if abs(skew_value) > 1
                else "Acceptable"
            )
        }
    )

skewness_summary = pd.DataFrame(
    skew_rows
)

save_csv(
    skewness_summary,
    "18_skewness_analysis.csv"
)


# ============================================================
# 45. COHORT STATISTICAL COMPARISON
# ============================================================

cohort_test_rows = []

cohort_groups = [
    group["retained_customer"].values
    for _, group in customer_features.groupby(
        "cohort_month_date"
    )
]

cohort_groups = [
    g
    for g in cohort_groups
    if len(g) >= 2
]

if len(cohort_groups) >= 2:

    try:

        h_stat, h_p = stats.kruskal(
            *cohort_groups
        )

        cohort_test_rows.append(
            {
                "test": "Kruskal-Wallis",
                "variable": "retained_customer",
                "statistic": h_stat,
                "p_value": h_p
            }
        )

    except Exception:
        pass

cohort_statistical_tests = pd.DataFrame(
    cohort_test_rows
)

save_csv(
    cohort_statistical_tests,
    "19_cohort_statistical_tests.csv"
)


# ============================================================
# 46. BEST / WORST COHORTS
# ============================================================

valid_cohorts = cohort_comparison[
    cohort_comparison["customers"] >= 10
].copy()

if not valid_cohorts.empty:

    best_cohort = valid_cohorts.loc[
        valid_cohorts["retained_rate_pct"].idxmax()
    ]

    worst_cohort = valid_cohorts.loc[
        valid_cohorts["retained_rate_pct"].idxmin()
    ]

else:

    best_cohort = None
    worst_cohort = None


# ============================================================
# 47. FINAL REPORT
# ============================================================

report_path = REPORT_DIR / "final_cohort_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "CUSTOMER COHORT RETENTION ANALYSIS\n"
    )

    report.write(
        "PART 1 — HISTORICAL RETENTION ANALYSIS\n"
    )

    report.write("=" * 80 + "\n\n")

    report.write("DATASET\n")
    report.write("-" * 80 + "\n")

    report.write(
        f"Dataset file: {DATA_PATH.name}\n"
    )

    report.write(
        f"Original rows: {initial_rows:,}\n"
    )

    report.write(
        f"Duplicate rows removed: "
        f"{duplicates_removed:,}\n"
    )

    report.write(
        f"Valid purchase rows: "
        f"{len(retention_df):,}\n"
    )

    report.write(
        f"Unique customers: "
        f"{total_customers:,}\n"
    )

    report.write(
        f"Unique orders: "
        f"{order_df[order_col].nunique():,}\n"
    )

    report.write(
        f"Acquisition cohorts: "
        f"{cohort_sizes.shape[0]:,}\n\n"
    )

    report.write("RETENTION DEFINITION\n")
    report.write("-" * 80 + "\n")

    report.write(
        "A retained customer is a customer who was active "
        "in more than one calendar month during the observed "
        "dataset period.\n"
    )

    report.write(
        "A non-retained customer was active in only one "
        "calendar month during the observed dataset period.\n\n"
    )

    report.write("OVERALL RETENTION\n")
    report.write("-" * 80 + "\n")

    report.write(
        f"Retained customers: "
        f"{retained_customers:,} "
        f"({retained_pct:.2f}%)\n"
    )

    report.write(
        f"Non-retained customers: "
        f"{non_retained_customers:,} "
        f"({non_retained_pct:.2f}%)\n\n"
    )

    report.write("IMPORTANT INTERPRETATION\n")
    report.write("-" * 80 + "\n")

    report.write(
        "Retention percentages describe observed historical "
        "behavior within the available dataset window. "
        "They should not be interpreted as causal effects "
        "or proof that a customer will never return after "
        "the observation period.\n\n"
    )

    if best_cohort is not None:

        report.write("BEST COHORT\n")
        report.write("-" * 80 + "\n")

        report.write(
            f"Cohort: "
            f"{best_cohort['cohort_month_date']}\n"
        )

        report.write(
            f"Customers: "
            f"{int(best_cohort['customers']):,}\n"
        )

        report.write(
            f"Observed retained rate: "
            f"{best_cohort['retained_rate_pct']:.2f}%\n\n"
        )

    if worst_cohort is not None:

        report.write("LOWEST COHORT\n")
        report.write("-" * 80 + "\n")

        report.write(
            f"Cohort: "
            f"{worst_cohort['cohort_month_date']}\n"
        )

        report.write(
            f"Customers: "
            f"{int(worst_cohort['customers']):,}\n"
        )

        report.write(
            f"Observed retained rate: "
            f"{worst_cohort['retained_rate_pct']:.2f}%\n\n"
        )

    report.write("KEY CUSTOMER BEHAVIOR\n")
    report.write("-" * 80 + "\n")

    if not retained_summary.empty:

        retained_row = retained_summary[
            retained_summary["retained_customer"] == 1
        ]

        non_retained_row = retained_summary[
            retained_summary["retained_customer"] == 0
        ]

        if not retained_row.empty and not non_retained_row.empty:

            report.write(
                f"Average orders — Retained: "
                f"{retained_row.iloc[0]['average_orders']:.2f}\n"
            )

            report.write(
                f"Average orders — Non-Retained: "
                f"{non_retained_row.iloc[0]['average_orders']:.2f}\n"
            )

            report.write(
                f"Average spend — Retained: "
                f"{retained_row.iloc[0]['average_spend']:.2f}\n"
            )

            report.write(
                f"Average spend — Non-Retained: "
                f"{non_retained_row.iloc[0]['average_spend']:.2f}\n"
            )

            report.write(
                f"Average active months — Retained: "
                f"{retained_row.iloc[0]['average_active_months']:.2f}\n"
            )

            report.write(
                f"Average active months — Non-Retained: "
                f"{non_retained_row.iloc[0]['average_active_months']:.2f}\n"
            )

    report.write("\nSTATISTICAL VALIDATION\n")
    report.write("-" * 80 + "\n")

    if not statistical_tests.empty:

        significant_features = statistical_tests[
            statistical_tests[
                "significant_after_fdr"
            ]
        ]["feature"].tolist()

        if significant_features:

            report.write(
                "Features showing statistically significant "
                "retained vs non-retained differences after "
                "FDR correction:\n"
            )

            for feature in significant_features:

                report.write(
                    f" - {feature}\n"
                )

        else:

            report.write(
                "No tested feature showed a statistically "
                "significant difference after FDR correction.\n"
            )

    report.write("\nOUTPUTS\n")
    report.write("-" * 80 + "\n")

    report.write(
        f"Tables: {TABLE_DIR}\n"
    )

    report.write(
        f"Plots: {PLOT_DIR}\n"
    )

    report.write(
        f"Report: {report_path}\n"
    )


# ============================================================
# 48. FINAL CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"Customers              : {total_customers:,}")
print(f"Retained               : {retained_customers:,}")
print(f"Non-retained           : {non_retained_customers:,}")
print(f"Retained %             : {retained_pct:.2f}%")
print(f"Non-retained %         : {non_retained_pct:.2f}%")
print(f"Cohorts                : {cohort_sizes.shape[0]:,}")
print(f"Orders                 : {order_df[order_col].nunique():,}")

print("\nOUTPUT DIRECTORIES")
print("-" * 80)
print(f"Tables : {TABLE_DIR}")
print(f"Plots  : {PLOT_DIR}")
print(f"Report : {REPORT_DIR}")

print("\nMAIN OUTPUTS")
print("-" * 80)

print("01_data_quality_summary.csv")
print("02_customer_summary.csv")
print("03_monthly_customer_activity.csv")
print("04_customer_features.csv")
print("05_cohort_sizes.csv")
print("06_cohort_retention_counts.csv")
print("07_cohort_retention_rates.csv")
print("08_cohort_retention_matrix.csv")
print("09_cohort_retention_count_matrix.csv")
print("10_retention_curve_summary.csv")
print("11_new_vs_returning_customers.csv")
print("12_retained_vs_nonretained_summary.csv")
print("13_statistical_tests.csv")
print("14_cohort_comparison.csv")
print("15_overall_retention_summary.csv")
print("17_customer_outlier_summary.csv")
print("18_skewness_analysis.csv")
print("19_cohort_statistical_tests.csv")

print("\nMAIN PLOTS")
print("-" * 80)

print("01_monthly_active_customers.png")
print("02_new_vs_returning_customers.png")
print("03_customer_order_distribution.png")
print("04_customer_spend_distribution.png")
print("05_cohort_size.png")
print("06_cohort_retention_heatmap.png")
print("07_retention_curves.png")
print("08_retained_vs_nonretained_comparison.png")
print("09_average_retention_curve.png")
print("10_retained_customer_balance.png")
print("11_customer_behavior_correlation.png")

print("\n" + "=" * 80)
print("PART 1 — CUSTOMER COHORT RETENTION ANALYSIS COMPLETE")
print("=" * 80)


#ML LEARNING MODELING



# ============================================================
# OLIST E-COMMERCE DATASET
# PROFESSIONAL + FAST BINARY CLASSIFICATION
# TARGET: delivery_late
#
# MODELS:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. Extra Trees
# 5. HistGradientBoosting
#
# OPTUNA:
# 100 trials per model
# 5 models x 100 = 500 total trials
#
# CV:
# 2-Fold Stratified Cross Validation
#
# OPTIMIZATION METRIC:
# ROC-AUC
#
# FINAL:
# - Protected Test Set
# - OOF Threshold Optimization
# - ROC Curve
# - Precision-Recall Curve
# - Confusion Matrix
# - Calibration Curve
# - Permutation Importance
# - Model Comparison
# - CSV/JSON/PKL outputs
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)

from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_TRIALS = 100

N_SPLITS = 2

N_JOBS = -1

MIN_CATEGORY_FREQUENCY = 10

# Maximum rows used for Optuna tuning
# Keeps execution faster while final model still uses
# complete training data.
TUNING_MAX_ROWS = 40_000

# Permutation importance sample
PERMUTATION_SAMPLE = 1_000

PERMUTATION_REPEATS = 1

THRESHOLD_MIN = 0.10

THRESHOLD_MAX = 0.90

THRESHOLD_STEP = 0.02

TARGET = "delivery_late"


# ============================================================
# 2. PATHS
# ============================================================

DATA_PATH = Path(r"C:\Users\Hp\Documents\New folder\olist_6_files_merged.csv")

OUTPUT_DIR = Path("ml_outputs")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. START TIMER
# ============================================================

TOTAL_START = time.time()


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH.resolve()}\n\n"
        "Please check DATA_PATH and CSV filename."
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded successfully.")
print(f"Shape: {df.shape}")


# ============================================================
# 5. TARGET VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

if TARGET not in df.columns:
    raise KeyError(
        f"Target column '{TARGET}' was not found in dataset."
    )

# Convert target to numeric
df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

# Remove missing target rows
before_target = len(df)

df = df.dropna(
    subset=[TARGET]
).copy()

after_target = len(df)

print(f"Rows removed because target was missing: "
      f"{before_target - after_target}")

# Convert to compact integer type
df[TARGET] = df[TARGET].astype("int8")

unique_target = sorted(
    df[TARGET].unique().tolist()
)

print(f"Target: {TARGET}")
print(f"Target values: {unique_target}")

if unique_target != [0, 1]:
    raise ValueError(
        f"'{TARGET}' must contain exactly binary values [0, 1]. "
        f"Found: {unique_target}"
    )

print("\nTarget distribution:")
print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

print("\nTarget percentage:")
print(
    (df[TARGET]
     .value_counts(normalize=True)
     .sort_index() * 100)
     .round(2)
)


# ============================================================
# 6. TARGET DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(7, 5))

df[TARGET].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Target Distribution - delivery_late")
plt.xlabel("delivery_late")
plt.ylabel("Count")
plt.xticks(
    rotation=0
)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_target_distribution.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 7. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)

    print(
        f"Duplicates removed. New shape: {df.shape}"
    )


# ============================================================
# 8. ID / IDENTIFIER COLUMNS
# ============================================================

ID_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "product_id",
    "seller_id",
    "review_id"
]

id_present = [
    c
    for c in ID_COLUMNS
    if c in df.columns
]


# ============================================================
# 9. POST-OUTCOME / LEAKAGE COLUMNS
# ============================================================

# IMPORTANT:
# TARGET MUST NOT be included here.
#
# delivery_late is our target and must remain in df_model
# until X and y are created.

POST_OUTCOME_COLUMNS = [
    "delivery_delay_days",
    "delivery_days",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "review_count",
    "avg_review_score",
    "review_comment_titles",
    "review_comment_messages"
]

leakage_present = [
    c
    for c in POST_OUTCOME_COLUMNS
    if c in df.columns
]


# ============================================================
# 10. DATETIME FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("DATETIME FEATURE ENGINEERING")
print("=" * 70)


def add_datetime_features(data):
    """
    Creates useful pre-outcome datetime features.

    Raw datetime columns are removed afterwards.

    Only information available around order placement/
    approval/estimated delivery is retained.
    """

    data = data.copy()

    purchase_col = "order_purchase_timestamp"
    approval_col = "order_approved_at"
    estimated_col = "order_estimated_delivery_date"

    purchase = None
    approval = None
    estimated = None

    # --------------------------------------------------------
    # Purchase timestamp
    # --------------------------------------------------------

    if purchase_col in data.columns:

        purchase = pd.to_datetime(
            data[purchase_col],
            errors="coerce"
        )

        data["purchase_year"] = (
            purchase.dt.year.astype("float32")
        )

        data["purchase_month"] = (
            purchase.dt.month.astype("float32")
        )

        data["purchase_day"] = (
            purchase.dt.day.astype("float32")
        )

        data["purchase_weekday"] = (
            purchase.dt.weekday.astype("float32")
        )

        data["purchase_hour"] = (
            purchase.dt.hour.astype("float32")
        )

        # Cyclic month
        data["purchase_month_sin"] = np.sin(
            2 * np.pi * data["purchase_month"] / 12
        ).astype("float32")

        data["purchase_month_cos"] = np.cos(
            2 * np.pi * data["purchase_month"] / 12
        ).astype("float32")

        # Cyclic weekday
        data["purchase_weekday_sin"] = np.sin(
            2 * np.pi * data["purchase_weekday"] / 7
        ).astype("float32")

        data["purchase_weekday_cos"] = np.cos(
            2 * np.pi * data["purchase_weekday"] / 7
        ).astype("float32")

        # Cyclic hour
        data["purchase_hour_sin"] = np.sin(
            2 * np.pi * data["purchase_hour"] / 24
        ).astype("float32")

        data["purchase_hour_cos"] = np.cos(
            2 * np.pi * data["purchase_hour"] / 24
        ).astype("float32")

    # --------------------------------------------------------
    # Approval timestamp
    # --------------------------------------------------------

    if approval_col in data.columns:

        approval = pd.to_datetime(
            data[approval_col],
            errors="coerce"
        )

        data["approval_weekday"] = (
            approval.dt.weekday.astype("float32")
        )

        data["approval_hour"] = (
            approval.dt.hour.astype("float32")
        )

    # --------------------------------------------------------
    # Purchase -> Approval time
    # --------------------------------------------------------

    if purchase is not None and approval is not None:

        data["purchase_to_approval_hours"] = (
            (
                approval - purchase
            )
            .dt.total_seconds()
            / 3600
        ).astype("float32")

    # --------------------------------------------------------
    # Estimated delivery date
    # --------------------------------------------------------

    if estimated_col in data.columns:

        estimated = pd.to_datetime(
            data[estimated_col],
            errors="coerce"
        )

        data["estimated_delivery_month"] = (
            estimated.dt.month.astype("float32")
        )

        data["estimated_delivery_weekday"] = (
            estimated.dt.weekday.astype("float32")
        )

    # --------------------------------------------------------
    # Purchase -> Estimated Delivery
    # --------------------------------------------------------

    if purchase is not None and estimated is not None:

        data["estimated_delivery_window_days"] = (
            (
                estimated - purchase
            )
            .dt.total_seconds()
            / (24 * 3600)
        ).astype("float32")

    # --------------------------------------------------------
    # Drop raw datetime columns
    # --------------------------------------------------------

    raw_datetime_columns = [
        purchase_col,
        approval_col,
        estimated_col
    ]

    raw_datetime_columns = [
        c
        for c in raw_datetime_columns
        if c in data.columns
    ]

    if raw_datetime_columns:

        data = data.drop(
            columns=raw_datetime_columns
        )

    return data


df_model = add_datetime_features(df)


# ============================================================
# 11. REMOVE IDS + LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("REMOVING IDS AND LEAKAGE")
print("=" * 70)

columns_to_drop = list(
    dict.fromkeys(
        id_present + leakage_present
    )
)

if columns_to_drop:

    df_model = df_model.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

print("\nID / identifier-like columns removed:")

for c in id_present:
    print(f"  - {c}")

print("\nPost-outcome / leakage columns removed:")

for c in leakage_present:
    print(f"  - {c}")


# ============================================================
# 12. AUTOMATIC SUSPICIOUS COLUMN REMOVAL
# ============================================================

print("\n" + "=" * 70)
print("AUTOMATIC LEAKAGE CHECK")
print("=" * 70)


SUSPICIOUS_PATTERNS = [
    "delivery_delay",
    "delivery_days",
    "delivered_",
    "review_",
    "review_score",
    "late_"
]

automatic_leakage = []

for col in df_model.columns:

    # NEVER remove target
    if col == TARGET:
        continue

    col_lower = col.lower()

    if any(
        pattern in col_lower
        for pattern in SUSPICIOUS_PATTERNS
    ):
        automatic_leakage.append(col)


if automatic_leakage:

    df_model = df_model.drop(
        columns=automatic_leakage,
        errors="ignore"
    )

    print("Additional suspicious columns removed:")

    for c in automatic_leakage:
        print(f"  - {c}")

else:

    print("No additional suspicious columns detected.")


# ============================================================
# 13. DATETIME-LIKE OBJECT CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATETIME-LIKE COLUMN CHECK")
print("=" * 70)

datetime_like_columns = []

for col in df_model.select_dtypes(
    include=["object"]
).columns:

    if col == TARGET:
        continue

    sample = (
        df_model[col]
        .dropna()
        .astype(str)
        .head(100)
    )

    if len(sample) == 0:
        continue

    converted = pd.to_datetime(
        sample,
        errors="coerce"
    )

    conversion_rate = converted.notna().mean()

    if conversion_rate >= 0.90:

        datetime_like_columns.append(col)


if datetime_like_columns:

    print(
        "Datetime-like object columns removed:"
    )

    for c in datetime_like_columns:
        print(f"  - {c}")

    df_model = df_model.drop(
        columns=datetime_like_columns,
        errors="ignore"
    )

else:

    print(
        "No remaining datetime-like object columns."
    )


# ============================================================
# 14. CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT FEATURE CHECK")
print("=" * 70)

constant_columns = [
    c
    for c in df_model.columns
    if c != TARGET
    and df_model[c].nunique(dropna=False) <= 1
]

print(
    f"Constant features found: "
    f"{len(constant_columns)}"
)

if constant_columns:

    for c in constant_columns:
        print(f"  - {c}")

    df_model = df_model.drop(
        columns=constant_columns
    )


# ============================================================
# 15. FINAL TARGET VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TARGET VALIDATION")
print("=" * 70)

if TARGET not in df_model.columns:

    raise KeyError(
        f"Target '{TARGET}' is missing from df_model.\n"
        "The target must NOT be removed during leakage cleaning."
    )

print(
    f"Target '{TARGET}' is correctly present."
)


# ============================================================
# 16. X / y
# ============================================================

print("\n" + "=" * 70)
print("CREATING X AND y")
print("=" * 70)

X = df_model.drop(
    columns=[TARGET]
).copy()

y = df_model[TARGET].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print(
    f"Target values: "
    f"{sorted(y.unique())}"
)


# ============================================================
# 17. NUMERICAL + CATEGORICAL FEATURES
# ============================================================

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print(
    f"Numerical features: "
    f"{len(numeric_columns)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_columns)}"
)


# ============================================================
# 18. HIGH-CARDINALITY CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CARDINALITY CATEGORICAL CHECK")
print("=" * 70)

high_cardinality_columns = []

for col in categorical_columns:

    unique_count = X[col].nunique(
        dropna=True
    )

    ratio = (
        unique_count / len(X)
        if len(X) > 0
        else 0
    )

    if ratio > 0.30:

        high_cardinality_columns.append(col)


if high_cardinality_columns:

    print(
        "High-cardinality categorical features removed:"
    )

    for c in high_cardinality_columns:
        print(f"  - {c}")

    X = X.drop(
        columns=high_cardinality_columns
    )

else:

    print(
        "No extreme high-cardinality categorical "
        "features found."
    )


# Recalculate feature types
numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ============================================================
# 19. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(
    f"Training rows: {len(X_train):,}"
)

print(
    f"Test rows: {len(X_test):,}"
)

print(
    f"Training percentage: "
    f"{len(X_train) / len(X) * 100:.1f}%"
)

print(
    f"Test percentage: "
    f"{len(X_test) / len(X) * 100:.1f}%"
)


# ============================================================
# 20. TUNING SUBSET
# ============================================================

print("\n" + "=" * 70)
print("CREATING TUNING SUBSET")
print("=" * 70)

if len(X_train) > TUNING_MAX_ROWS:

    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=TUNING_MAX_ROWS,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

else:

    X_tune = X_train.copy()
    y_tune = y_train.copy()


print(
    f"Tuning rows: {len(X_tune):,}"
)


# ============================================================
# 21. PREPROCESSOR HELPERS
# ============================================================

def make_onehot_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_CATEGORY_FREQUENCY,
            sparse_output=True,
            dtype=np.float32
        )

    except TypeError:

        # Compatibility with older scikit-learn
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=MIN_CATEGORY_FREQUENCY,
            sparse=True,
            dtype=np.float32
        )


def make_logistic_preprocessor(
    numerical_features,
    categorical_features
):

    transformers = []

    if numerical_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "num",
                numeric_pipeline,
                numerical_features
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    make_onehot_encoder()
                )
            ]
        )

        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def make_tree_preprocessor(
    numerical_features,
    categorical_features
):

    transformers = []

    if numerical_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        transformers.append(
            (
                "num",
                numeric_pipeline,
                numerical_features
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        dtype=np.float32
                    )
                )
            ]
        )

        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# 22. MODEL SEARCH SPACES
# ============================================================

def suggest_params(
    trial,
    model_name
):

    if model_name == "Logistic Regression":

        return {
            "C": trial.suggest_float(
                "C",
                0.03,
                3.0,
                log=True
            ),
            "penalty": trial.suggest_categorical(
                "penalty",
                ["l1", "l2"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"]
            )
        }

    elif model_name == "Decision Tree":

        return {
            "max_depth": trial.suggest_int(
                "max_depth",
                4,
                14
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                15
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                10
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                [None, "sqrt"]
            ),
            "criterion": trial.suggest_categorical(
                "criterion",
                ["gini", "entropy"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"]
            )
        }

    elif model_name == "Random Forest":

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                40,
                100,
                step=20
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                5,
                14
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                10
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                5
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"]
            )
        }

    elif model_name == "Extra Trees":

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                40,
                100,
                step=20
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                5,
                14
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                10
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                5
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"]
            )
        }

    elif model_name == "HistGradientBoosting":

        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.04,
                0.15,
                log=True
            ),
            "max_iter": trial.suggest_int(
                "max_iter",
                60,
                140,
                step=20
            ),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes",
                15,
                35
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                8
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                15,
                50
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization",
                0.001,
                3.0,
                log=True
            )
        }

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# 23. MODEL FACTORY
# ============================================================

def create_model(
    model_name,
    params
):

    if model_name == "Logistic Regression":

        return LogisticRegression(
            **params,
            solver="liblinear",
            max_iter=150,
            tol=1e-3,
            random_state=RANDOM_STATE
        )

    elif model_name == "Decision Tree":

        return DecisionTreeClassifier(
            **params,
            random_state=RANDOM_STATE
        )

    elif model_name == "Random Forest":

        return RandomForestClassifier(
            **params,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        )

    elif model_name == "Extra Trees":

        return ExtraTreesClassifier(
            **params,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        )

    elif model_name == "HistGradientBoosting":

        return HistGradientBoostingClassifier(
            **params,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.10,
            random_state=RANDOM_STATE
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# 24. MODEL LIST
# ============================================================

MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "HistGradientBoosting"
]


# ============================================================
# 25. PREPARE TUNING FOLDS
# ============================================================

print("\n" + "=" * 70)
print("PREPARING TUNING FOLDS")
print("=" * 70)

skf_tuning = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

tuning_folds = list(
    skf_tuning.split(
        X_tune,
        y_tune
    )
)

print(
    f"Tuning folds prepared: {N_SPLITS}"
)


# ============================================================
# 26. OPTUNA STORAGE
# ============================================================

studies = {}

best_params_dict = {}

optimization_summary = {}

all_trial_records = {}


# ============================================================
# 27. OPTUNA MODEL OPTIMIZATION
# ============================================================

def optimize_model(
    model_name
):

    print("\n")
    print("#" * 70)
    print(
        f"OPTIMIZING: {model_name}"
    )
    print(
        f"Trials: {N_TRIALS}"
    )
    print(
        f"CV folds: {N_SPLITS}"
    )
    print("#" * 70)

    model_start = time.time()

    # --------------------------------------------------------
    # Cache preprocessed fold data
    # --------------------------------------------------------

    cached_folds = []

    for fold_number, (
        train_idx,
        valid_idx
    ) in enumerate(tuning_folds, start=1):

        X_fold_train = X_tune.iloc[
            train_idx
        ]

        X_fold_valid = X_tune.iloc[
            valid_idx
        ]

        y_fold_train = y_tune.iloc[
            train_idx
        ]

        y_fold_valid = y_tune.iloc[
            valid_idx
        ]

        # ----------------------------------------------------
        # Fit preprocessing ONLY on fold training data
        # ----------------------------------------------------

        if model_name == "Logistic Regression":

            preprocessor = make_logistic_preprocessor(
                numeric_columns,
                categorical_columns
            )

        else:

            preprocessor = make_tree_preprocessor(
                numeric_columns,
                categorical_columns
            )

        X_fold_train_processed = (
            preprocessor.fit_transform(
                X_fold_train
            )
        )

        X_fold_valid_processed = (
            preprocessor.transform(
                X_fold_valid
            )
        )

        cached_folds.append(
            (
                X_fold_train_processed,
                X_fold_valid_processed,
                y_fold_train.to_numpy(),
                y_fold_valid.to_numpy()
            )
        )

        del (
            X_fold_train,
            X_fold_valid,
            y_fold_train,
            y_fold_valid,
            preprocessor
        )

        gc.collect()

        print(
            f"Prepared fold {fold_number}/{N_SPLITS}"
        )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(trial):

        params = suggest_params(
            trial,
            model_name
        )

        fold_scores = []

        for fold_number, (
            X_tr,
            X_val,
            y_tr,
            y_val
        ) in enumerate(
            cached_folds,
            start=1
        ):

            model = create_model(
                model_name,
                params
            )

            try:

                model.fit(
                    X_tr,
                    y_tr
                )

                probabilities = model.predict_proba(
                    X_val
                )[:, 1]

                score = roc_auc_score(
                    y_val,
                    probabilities
                )

                fold_scores.append(
                    score
                )

                running_score = float(
                    np.mean(fold_scores)
                )

                trial.report(
                    running_score,
                    step=fold_number
                )

                if trial.should_prune():

                    raise optuna.TrialPruned()

            except Exception as e:

                raise optuna.TrialPruned(
                    f"Trial failed: {e}"
                )

            finally:

                del model

        return float(
            np.mean(fold_scores)
        )

    # --------------------------------------------------------
    # Optuna Study
    # --------------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=model_name.replace(
            " ",
            "_"
        )
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True
    )

    elapsed = time.time() - model_start

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    studies[model_name] = study

    best_params_dict[model_name] = (
        study.best_params
    )

    optimization_summary[model_name] = {
        "best_cv_roc_auc": float(
            study.best_value
        ),
        "best_trial": int(
            study.best_trial.number
        ),
        "completed_trials": len(
            study.trials
        ),
        "optimization_time_seconds": float(
            elapsed
        )
    }

    trial_records = []

    for trial in study.trials:

        trial_records.append(
            {
                "model": model_name,
                "trial_number": trial.number,
                "state": str(trial.state),
                "value": (
                    float(trial.value)
                    if trial.value is not None
                    else np.nan
                ),
                **trial.params
            }
        )

    all_trial_records[
        model_name
    ] = pd.DataFrame(
        trial_records
    )

    print(
        f"\n{model_name} completed."
    )

    print(
        f"Best CV ROC-AUC: "
        f"{study.best_value:.6f}"
    )

    print(
        f"Best trial: "
        f"{study.best_trial.number}"
    )

    print(
        f"Time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "Best parameters:"
    )

    for key, value in study.best_params.items():
        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Free cached data
    # --------------------------------------------------------

    del cached_folds

    gc.collect()

    return study


# ============================================================
# 28. RUN ALL 5 MODELS
# ============================================================

for model_name in MODELS:

    optimize_model(
        model_name
    )

    gc.collect()


# ============================================================
# 29. SAVE OPTUNA TRIAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING OPTUNA RESULTS")
print("=" * 70)

for model_name, trials_df in all_trial_records.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    trials_df.to_csv(
        OUTPUT_DIR
        / f"optuna_trials_{safe_name}.csv",
        index=False
    )


# ============================================================
# 30. SAVE OPTUNA SUMMARY
# ============================================================

optimization_df = pd.DataFrame(
    [
        {
            "Model": model_name,
            **optimization_summary[model_name]
        }
        for model_name in MODELS
    ]
)

optimization_df = optimization_df.sort_values(
    "best_cv_roc_auc",
    ascending=False
)

optimization_df.to_csv(
    OUTPUT_DIR
    / "optuna_summary.csv",
    index=False
)


# ============================================================
# 31. FINAL MODEL TRAINING
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)


final_models = {}

test_probabilities = {}

test_predictions_05 = {}

model_results = []


for model_name in MODELS:

    print("\n" + "-" * 70)
    print(
        f"FINAL TRAINING: {model_name}"
    )
    print("-" * 70)

    model_start = time.time()

    # --------------------------------------------------------
    # Create final preprocessor
    # --------------------------------------------------------

    if model_name == "Logistic Regression":

        preprocessor = make_logistic_preprocessor(
            numeric_columns,
            categorical_columns
        )

    else:

        preprocessor = make_tree_preprocessor(
            numeric_columns,
            categorical_columns
        )

    # --------------------------------------------------------
    # Fit preprocessor ONLY on training data
    # --------------------------------------------------------

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        f"Processed train shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed test shape: "
        f"{X_test_processed.shape}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    best_params = (
        best_params_dict[model_name]
    )

    final_model = create_model(
        model_name,
        best_params
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    final_model.fit(
        X_train_processed,
        y_train
    )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    probabilities = (
        final_model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype("int8")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    train_time = time.time() - model_start

    model_results.append(
        {
            "Model": model_name,
            "Best CV ROC-AUC": (
                optimization_summary[
                    model_name
                ]["best_cv_roc_auc"]
            ),
            "Test Accuracy": accuracy,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1": f1,
            "Test ROC-AUC": roc_auc,
            "Test PR-AUC": pr_auc,
            "Test Brier Score": brier,
            "Training Time Seconds": train_time
        }
    )

    final_models[
        model_name
    ] = {
        "preprocessor": preprocessor,
        "model": final_model
    }

    test_probabilities[
        model_name
    ] = probabilities

    test_predictions_05[
        model_name
    ] = predictions

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    print(
        f"Time     : {train_time:.2f} sec"
    )

    del (
        X_train_processed,
        X_test_processed
    )

    gc.collect()


# ============================================================
# 32. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    model_results
)

results_df = results_df.sort_values(
    "Best CV ROC-AUC",
    ascending=False
).reset_index(drop=True)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    OUTPUT_DIR
    / "model_comparison.csv",
    index=False
)


# ============================================================
# 33. BEST MODEL SELECTION
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best model based on CV ROC-AUC: "
    f"{best_model_name}"
)

print(
    f"Best CV ROC-AUC: "
    f"{results_df.iloc[0]['Best CV ROC-AUC']:.6f}"
)

print(
    f"Test ROC-AUC: "
    f"{results_df.iloc[0]['Test ROC-AUC']:.6f}"
)


# ============================================================
# 34. OOF PREDICTIONS FOR THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("OOF THRESHOLD OPTIMIZATION")
print("=" * 70)


best_params = (
    best_params_dict[
        best_model_name
    ]
)

oof_probabilities = np.zeros(
    len(X_train),
    dtype=np.float32
)

skf_oof = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


for fold_number, (
    train_idx,
    valid_idx
) in enumerate(
    skf_oof.split(
        X_train,
        y_train
    ),
    start=1
):

    print(
        f"\nOOF Fold "
        f"{fold_number}/{N_SPLITS}"
    )

    X_fold_train = X_train.iloc[
        train_idx
    ]

    X_fold_valid = X_train.iloc[
        valid_idx
    ]

    y_fold_train = y_train.iloc[
        train_idx
    ]

    # --------------------------------------------------------
    # Preprocessing inside fold
    # --------------------------------------------------------

    if best_model_name == "Logistic Regression":

        oof_preprocessor = (
            make_logistic_preprocessor(
                numeric_columns,
                categorical_columns
            )
        )

    else:

        oof_preprocessor = (
            make_tree_preprocessor(
                numeric_columns,
                categorical_columns
            )
        )

    X_fold_train_processed = (
        oof_preprocessor.fit_transform(
            X_fold_train
        )
    )

    X_fold_valid_processed = (
        oof_preprocessor.transform(
            X_fold_valid
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    oof_model = create_model(
        best_model_name,
        best_params
    )

    oof_model.fit(
        X_fold_train_processed,
        y_fold_train
    )

    fold_probabilities = (
        oof_model.predict_proba(
            X_fold_valid_processed
        )[:, 1]
    )

    oof_probabilities[
        valid_idx
    ] = fold_probabilities

    del (
        X_fold_train,
        X_fold_valid,
        y_fold_train,
        X_fold_train_processed,
        X_fold_valid_processed,
        oof_preprocessor,
        oof_model
    )

    gc.collect()


# ============================================================
# 35. FIND BEST THRESHOLD
# ============================================================

thresholds = np.arange(
    THRESHOLD_MIN,
    THRESHOLD_MAX + 0.001,
    THRESHOLD_STEP
)

threshold_records = []

best_threshold = 0.50

best_threshold_f1 = -1

for threshold in thresholds:

    oof_predictions = (
        oof_probabilities >= threshold
    ).astype("int8")

    f1 = f1_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    precision = precision_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_train,
        oof_predictions
    )

    threshold_records.append(
        {
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )

    if f1 > best_threshold_f1:

        best_threshold_f1 = f1

        best_threshold = threshold


threshold_df = pd.DataFrame(
    threshold_records
)

threshold_df.to_csv(
    OUTPUT_DIR
    / "threshold_optimization.csv",
    index=False
)

print(
    f"\nBest OOF threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best OOF F1: "
    f"{best_threshold_f1:.4f}"
)


# ============================================================
# 36. THRESHOLD PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    label="F1"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker=".",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker=".",
    label="Recall"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold = {best_threshold:.2f}"
)

plt.title(
    f"Threshold Optimization - {best_model_name}"
)

plt.xlabel("Threshold")
plt.ylabel("Score")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "02_threshold_optimization.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 37. FINAL BEST MODEL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL BEST MODEL TEST EVALUATION")
print("=" * 70)

best_test_probabilities = (
    test_probabilities[
        best_model_name
    ]
)

# Default threshold
best_test_predictions_05 = (
    best_test_probabilities >= 0.50
).astype("int8")

# Optimized threshold
best_test_predictions_optimal = (
    best_test_probabilities >= best_threshold
).astype("int8")


def calculate_metrics(
    y_true,
    predictions,
    probabilities
):

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities
        ),
        "PR-AUC": average_precision_score(
            y_true,
            probabilities
        ),
        "Brier Score": brier_score_loss(
            y_true,
            probabilities
        )
    }


default_metrics = calculate_metrics(
    y_test,
    best_test_predictions_05,
    best_test_probabilities
)

optimal_metrics = calculate_metrics(
    y_test,
    best_test_predictions_optimal,
    best_test_probabilities
)


print("\nDefault threshold = 0.50")

for key, value in default_metrics.items():

    print(
        f"{key}: {value:.4f}"
    )


print(
    f"\nOptimized threshold = "
    f"{best_threshold:.2f}"
)

for key, value in optimal_metrics.items():

    print(
        f"{key}: {value:.4f}"
    )


# ============================================================
# 38. FINAL METRICS TABLE
# ============================================================

final_metrics_df = pd.DataFrame(
    [
        {
            "Model": best_model_name,
            "Threshold Type": "Default",
            "Threshold": 0.50,
            **default_metrics
        },
        {
            "Model": best_model_name,
            "Threshold Type": "OOF Optimized",
            "Threshold": best_threshold,
            **optimal_metrics
        }
    ]
)

final_metrics_df.to_csv(
    OUTPUT_DIR
    / "final_best_model_metrics.csv",
    index=False
)


# ============================================================
# 39. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        best_test_predictions_optimal,
        digits=4,
        zero_division=0
    )
)

classification_report_dict = (
    classification_report(
        y_test,
        best_test_predictions_optimal,
        output_dict=True,
        zero_division=0
    )
)

classification_report_df = pd.DataFrame(
    classification_report_dict
).transpose()

classification_report_df.to_csv(
    OUTPUT_DIR
    / "classification_report.csv"
)


# ============================================================
# 40. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_test_predictions_optimal
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["0", "1"]
)

plt.yticks(
    [0, 1],
    ["0", "1"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "03_confusion_matrix.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 41. ROC CURVES - ALL MODELS
# ============================================================

print("\n" + "=" * 70)
print("ROC CURVES")
print("=" * 70)

plt.figure(figsize=(9, 7))

for model_name in MODELS:

    probabilities = (
        test_probabilities[
            model_name
        ]
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc_score:.4f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.title(
    "ROC Curves - All Models"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_roc_curves_all_models.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 42. PRECISION-RECALL CURVES
# ============================================================

print("\n" + "=" * 70)
print("PRECISION-RECALL CURVES")
print("=" * 70)

plt.figure(figsize=(9, 7))

for model_name in MODELS:

    probabilities = (
        test_probabilities[
            model_name
        ]
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_test,
            probabilities
        )
    )

    ap_score = average_precision_score(
        y_test,
        probabilities
    )

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"{model_name} (AP={ap_score:.4f})"
    )


plt.title(
    "Precision-Recall Curves - All Models"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "05_precision_recall_curves.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 43. MODEL METRIC COMPARISON
# ============================================================

metric_columns = [
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1",
    "Test ROC-AUC",
    "Test PR-AUC"
]


for metric in metric_columns:

    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Model"],
        results_df[metric]
    )

    plt.title(
        metric + " - Model Comparison"
    )

    plt.ylabel(
        metric
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    safe_metric = (
        metric
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        OUTPUT_DIR
        / f"metric_{safe_metric}.png",
        dpi=150
    )

    plt.show()

    plt.close()


# ============================================================
# 44. CALIBRATION CURVE
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION CURVE")
print("=" * 70)

prob_true, prob_pred = calibration_curve(
    y_test,
    best_test_probabilities,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(8, 6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label=best_model_name
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.title(
    f"Calibration Curve - {best_model_name}"
)

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Fraction of Positives"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "06_calibration_curve.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 45. PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE")
print("=" * 70)

best_pipeline_data = (
    final_models[
        best_model_name
    ]
)

best_preprocessor = (
    best_pipeline_data[
        "preprocessor"
    ]
)

best_model = (
    best_pipeline_data[
        "model"
    ]
)

# ------------------------------------------------------------
# Sample test data for speed
# ------------------------------------------------------------

permutation_size = min(
    PERMUTATION_SAMPLE,
    len(X_test)
)

X_perm = X_test.sample(
    n=permutation_size,
    random_state=RANDOM_STATE
)

y_perm = y_test.loc[
    X_perm.index
]

X_perm_processed = (
    best_preprocessor.transform(
        X_perm
    )
)

# ------------------------------------------------------------
# Permutation importance
# ------------------------------------------------------------

importance_result = permutation_importance(
    best_model,
    X_perm_processed,
    y_perm,
    scoring="roc_auc",
    n_repeats=PERMUTATION_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

# ------------------------------------------------------------
# Feature names
# ------------------------------------------------------------

try:

    feature_names = (
        best_preprocessor
        .get_feature_names_out()
    )

except Exception:

    feature_names = np.array(
        [
            f"feature_{i}"
            for i in range(
                len(
                    importance_result.importances_mean
                )
            )
        ]
    )


importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance Mean":
            importance_result.importances_mean,
        "Importance Std":
            importance_result.importances_std
    }
)

importance_df = (
    importance_df
    .sort_values(
        "Importance Mean",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_df.to_csv(
    OUTPUT_DIR
    / "permutation_importance.csv",
    index=False
)

print(
    "\nTop 20 features:"
)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 46. TOP FEATURE IMPORTANCE PLOT
# ============================================================

top_n = min(
    20,
    len(importance_df)
)

top_features = (
    importance_df
    .head(top_n)
    .sort_values(
        "Importance Mean"
    )
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance Mean"]
)

plt.title(
    f"Top {top_n} Permutation Features - "
    f"{best_model_name}"
)

plt.xlabel(
    "Mean ROC-AUC Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "07_top_permutation_importance.png",
    dpi=150
)

plt.show()

plt.close()


# ============================================================
# 47. SAVE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAVING TEST PREDICTIONS")
print("=" * 70)

test_predictions_df = pd.DataFrame(
    {
        "Actual": y_test.to_numpy(),
        "Probability": best_test_probabilities,
        "Prediction_0.50": (
            best_test_predictions_05
        ),
        "Prediction_Optimal": (
            best_test_predictions_optimal
        )
    }
)

test_predictions_df.to_csv(
    OUTPUT_DIR
    / "best_model_test_predictions.csv",
    index=False
)


# ============================================================
# 48. SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING BEST MODEL")
print("=" * 70)

best_model_path = (
    OUTPUT_DIR
    / "best_model_pipeline.pkl"
)

joblib.dump(
    final_models[
        best_model_name
    ],
    best_model_path
)

print(
    f"Saved: {best_model_path}"
)


# ============================================================
# 49. SAVE BEST PARAMETERS
# ============================================================

best_params_output = {
    model_name: {
        str(k): (
            None
            if v is None
            else (
                bool(v)
                if isinstance(v, np.bool_)
                else (
                    v.item()
                    if isinstance(v, np.generic)
                    else v
                )
            )
        )
        for k, v in params.items()
    }
    for model_name, params
    in best_params_dict.items()
}

with open(
    OUTPUT_DIR / "best_parameters.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        best_params_output,
        f,
        indent=4
    )


# ============================================================
# 50. SAVE CONFIGURATION
# ============================================================

config = {
    "target": TARGET,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "n_trials_per_model": N_TRIALS,
    "number_of_models": len(MODELS),
    "total_optuna_trials_requested":
        N_TRIALS * len(MODELS),
    "cv_folds": N_SPLITS,
    "tuning_max_rows": TUNING_MAX_ROWS,
    "threshold_min": THRESHOLD_MIN,
    "threshold_max": THRESHOLD_MAX,
    "threshold_step": THRESHOLD_STEP,
    "best_model": best_model_name,
    "best_threshold": float(best_threshold),
    "dataset_shape": list(df.shape),
    "X_shape": list(X.shape),
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape)
}

with open(
    OUTPUT_DIR / "config.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# 51. FINAL SUMMARY
# ============================================================

total_time = (
    time.time()
    - TOTAL_START
)

print("\n")
print("=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)

print(
    f"Dataset shape: "
    f"{df.shape}"
)

print(
    f"Features used: "
    f"{X.shape[1]}"
)

print(
    f"Training rows: "
    f"{len(X_train):,}"
)

print(
    f"Testing rows: "
    f"{len(X_test):,}"
)

print(
    f"Models: "
    f"{len(MODELS)}"
)

print(
    f"Optuna trials/model: "
    f"{N_TRIALS}"
)

print(
    f"Requested total trials: "
    f"{N_TRIALS * len(MODELS)}"
)

print(
    f"CV folds: "
    f"{N_SPLITS}"
)

print(
    f"Best model: "
    f"{best_model_name}"
)

print(
    f"Best OOF threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best CV ROC-AUC: "
    f"{results_df.iloc[0]['Best CV ROC-AUC']:.6f}"
)

print(
    f"Best model Test ROC-AUC: "
    f"{optimal_metrics['ROC-AUC']:.6f}"
)

print(
    f"Best model Test F1: "
    f"{optimal_metrics['F1']:.6f}"
)

print(
    f"Total execution time: "
    f"{total_time / 60:.2f} minutes"
)

print(
    f"\nOutput folder:"
)

print(
    OUTPUT_DIR.resolve()
)

print("\nGenerated files include:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  - {file.name}"
    )

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)


# ============================================================
# END
# ============================================================
