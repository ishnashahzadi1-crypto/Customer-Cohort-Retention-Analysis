# ============================================================
# CUSTOMER COHORT RETENTION ANALYSIS
# PART 1 — PROFESSIONAL DATA ANALYTICS PIPELINE
#
# Covers:
# 1. Data Understanding
# 2. Data Quality Assessment
# 3. Data Cleaning
# 4. EDA
# 5. Customer-Level Feature Engineering
# 6. RFM Analysis
# 7. Cohort Analysis
# 8. Retention Matrix
# 9. Repeat vs One-Time Customers
# 10. Outlier Analysis
# 11. Statistical Analysis
# 12. Customer Segmentation
# 13. Cohort Comparison
# 14. Business Insights
#
# PART 2 — ML / Prediction is intentionally NOT included.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

from pathlib import Path
import warnings
import logging
import sys
import json
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Professional plotting defaults
sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["figure.dpi"] = 120


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

# ============================================================
# CHANGE ONLY THIS PATH
# ============================================================

DATA_PATH = Path(
    r"Customer.py/olist_customer_cohort_merged.csv"
)

# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "part1"

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "reports"

for directory in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOGGING
# ============================================================

LOG_FILE = REPORT_DIR / "part1_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("customer_retention")


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def section(title):
    logger.info("")
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def save_csv(df, filename):
    path = TABLE_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")
    return path


def save_figure(filename):
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {path}")
    return path


def normalize_columns(df):
    """
    Normalize column names while preserving readability.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def find_column(df, candidates, required=False):
    """
    Find a column using exact names first and normalized matching second.
    """

    columns = set(df.columns)

    # Exact
    for candidate in candidates:
        candidate = candidate.lower()
        if candidate in columns:
            return candidate

    # Partial / fuzzy
    for column in df.columns:
        for candidate in candidates:
            candidate = candidate.lower()

            if candidate in column:
                return column

    if required:
        raise ValueError(
            f"Required column not found.\n"
            f"Tried: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def month_difference(start, end):
    """
    Difference in calendar months.
    """
    return (
        (end.dt.year - start.dt.year) * 12
        + (end.dt.month - start.dt.month)
    )


def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper


def cramers_v(table):
    """
    Calculate Cramer's V.
    """
    chi2 = stats.chi2_contingency(table)[0]

    n = table.sum().sum()

    if n == 0:
        return np.nan

    phi2 = chi2 / n

    r, k = table.shape

    phi2corr = max(
        0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1)
    )

    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(kcorr - 1, rcorr - 1)

    if denominator <= 0:
        return np.nan

    return np.sqrt(phi2corr / denominator)


# ============================================================
# 4. LOAD DATA
# ============================================================

section("STEP 1 — LOADING DATA")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        f"Change DATA_PATH at the top of the script."
    )

logger.info(f"Reading dataset: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

logger.info(f"Initial shape: {df.shape}")

df = normalize_columns(df)

logger.info("Column names normalized.")


# ============================================================
# 5. INITIAL DATA UNDERSTANDING
# ============================================================

section("STEP 2 — DATA UNDERSTANDING")

initial_rows = len(df)
initial_columns = len(df.columns)

memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

logger.info(f"Rows: {initial_rows:,}")
logger.info(f"Columns: {initial_columns:,}")
logger.info(f"Memory usage: {memory_mb:.2f} MB")


# ============================================================
# 6. COLUMN DETECTION
# ============================================================

section("STEP 3 — DETECTING IMPORTANT COLUMNS")

customer_id_col = find_column(
    df,
    [
        "customer_unique_id",
        "customer_id",
        "customerid"
    ],
    required=True
)

order_id_col = find_column(
    df,
    [
        "order_id",
        "orderid"
    ],
    required=True
)

order_date_col = find_column(
    df,
    [
        "order_purchase_timestamp",
        "purchase_timestamp",
        "purchase_date",
        "order_date",
        "purchase_datetime"
    ],
    required=True
)

price_col = find_column(
    df,
    [
        "price",
        "total_price",
        "payment_value"
    ],
    required=False
)

freight_col = find_column(
    df,
    [
        "freight_value",
        "shipping_cost"
    ],
    required=False
)

product_col = find_column(
    df,
    [
        "product_id"
    ],
    required=False
)

category_col = find_column(
    df,
    [
        "product_category_name",
        "category_name",
        "product_category"
    ],
    required=False
)

seller_col = find_column(
    df,
    [
        "seller_id"
    ],
    required=False
)

logger.info(f"Customer ID: {customer_id_col}")
logger.info(f"Order ID: {order_id_col}")
logger.info(f"Purchase date: {order_date_col}")
logger.info(f"Price: {price_col}")
logger.info(f"Freight: {freight_col}")
logger.info(f"Product: {product_col}")
logger.info(f"Category: {category_col}")
logger.info(f"Seller: {seller_col}")


# ============================================================
# 7. DATA QUALITY SUMMARY — BEFORE CLEANING
# ============================================================

section("STEP 4 — DATA QUALITY ASSESSMENT")

missing_count = df.isna().sum()

missing_pct = (
    missing_count / len(df) * 100
)

duplicate_rows = df.duplicated().sum()

infinite_count = np.isinf(
    df.select_dtypes(include=np.number)
).sum().sum()

quality_summary = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing_count": missing_count.values,
    "missing_percentage": missing_pct.values,
    "unique_values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ],
    "constant_column": [
        df[col].nunique(dropna=False) <= 1
        for col in df.columns
    ]
})

quality_summary["potential_id_like"] = (
    quality_summary["unique_values"] / len(df) >= 0.95
)

save_csv(
    quality_summary,
    "01_data_quality_summary.csv"
)

logger.info(f"Duplicate rows: {duplicate_rows:,}")
logger.info(f"Infinite numeric values: {infinite_count:,}")


# ============================================================
# 8. DATE CONVERSION
# ============================================================

section("STEP 5 — DATE PROCESSING")

df[order_date_col] = pd.to_datetime(
    df[order_date_col],
    errors="coerce"
)

invalid_dates = df[order_date_col].isna().sum()

logger.info(f"Invalid/missing purchase dates: {invalid_dates:,}")


# ============================================================
# 9. NUMERIC CLEANING
# ============================================================

section("STEP 6 — NUMERIC CLEANING")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

numeric_summary = []

for col in numeric_columns:

    series = safe_numeric(df[col])

    numeric_summary.append({
        "column": col,
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "missing": series.isna().sum(),
        "zeros": (series == 0).sum(),
        "negative_values": (series < 0).sum()
    })

numeric_summary = pd.DataFrame(numeric_summary)

save_csv(
    numeric_summary,
    "02_numerical_statistics_before_cleaning.csv"
)


# ============================================================
# 10. REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

removed_duplicates = before_duplicates - len(df)

logger.info(
    f"Exact duplicate rows removed: {removed_duplicates:,}"
)


# ============================================================
# 11. HANDLE INFINITE VALUES
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns

if len(numeric_cols) > 0:

    df[numeric_cols] = df[numeric_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# 12. BASIC INVALID VALUE CLEANING
# ============================================================

if price_col:

    df[price_col] = safe_numeric(df[price_col])

    negative_prices = (
        df[price_col] < 0
    ).sum()

    logger.info(
        f"Negative price values: {negative_prices:,}"
    )

    if negative_prices > 0:
        df.loc[
            df[price_col] < 0,
            price_col
        ] = np.nan


if freight_col:

    df[freight_col] = safe_numeric(
        df[freight_col]
    )

    negative_freight = (
        df[freight_col] < 0
    ).sum()

    logger.info(
        f"Negative freight values: {negative_freight:,}"
    )

    if negative_freight > 0:
        df.loc[
            df[freight_col] < 0,
            freight_col
        ] = np.nan


# ============================================================
# 13. DROP RECORDS WITHOUT CRITICAL IDENTIFIERS/DATES
# ============================================================

critical_before = len(df)

df = df.dropna(
    subset=[
        customer_id_col,
        order_id_col,
        order_date_col
    ]
)

critical_removed = critical_before - len(df)

logger.info(
    f"Rows removed due to missing critical fields: "
    f"{critical_removed:,}"
)


# ============================================================
# 14. CUSTOMER-ORDER LEVEL DATASET
# ============================================================

section("STEP 7 — BUILDING CUSTOMER-ORDER DATA")

# Avoid counting duplicated merged rows as separate orders.
#
# We aggregate transaction-level information to order level first.

order_group_cols = [
    order_id_col,
    customer_id_col
]

order_agg = {
    order_date_col: "min"
}

if price_col:
    order_agg[price_col] = "sum"

if freight_col:
    order_agg[freight_col] = "sum"

if product_col:
    order_agg[product_col] = "nunique"

if category_col:
    order_agg[category_col] = "nunique"

if seller_col:
    order_agg[seller_col] = "nunique"


order_df = (
    df.groupby(
        order_group_cols,
        as_index=False
    )
    .agg(order_agg)
)


# Rename derived columns

rename_map = {}

if product_col:
    rename_map[product_col] = "unique_products_per_order"

if category_col:
    rename_map[category_col] = "unique_categories_per_order"

if seller_col:
    rename_map[seller_col] = "unique_sellers_per_order"

order_df = order_df.rename(
    columns=rename_map
)

logger.info(
    f"Customer-order dataset shape: {order_df.shape}"
)


# ============================================================
# 15. ORDER VALUE
# ============================================================

if price_col and freight_col:

    order_df["order_value"] = (
        order_df[price_col].fillna(0)
        + order_df[freight_col].fillna(0)
    )

elif price_col:

    order_df["order_value"] = (
        order_df[price_col]
    )

elif freight_col:

    order_df["order_value"] = (
        order_df[freight_col]
    )

else:

    raise ValueError(
        "No usable monetary column detected. "
        "A price/payment column is required "
        "for monetary/RFM analysis."
    )


# ============================================================
# 16. CUSTOMER-LEVEL FEATURE ENGINEERING
# ============================================================

section("STEP 8 — CUSTOMER-LEVEL FEATURE ENGINEERING")

customer_agg = {
    order_id_col: "nunique",
    order_date_col: ["min", "max"],
    "order_value": [
        "sum",
        "mean",
        "min",
        "max"
    ]
}

customer_df = (
    order_df
    .groupby(
        customer_id_col
    )
    .agg(customer_agg)
)

customer_df.columns = [
    "_".join(
        [str(x) for x in col if str(x) != ""]
    ).strip("_")
    for col in customer_df.columns
]

customer_df = customer_df.reset_index()


customer_df = customer_df.rename(
    columns={
        f"{order_id_col}_nunique": "total_orders",
        f"{order_date_col}_min": "first_purchase_date",
        f"{order_date_col}_max": "last_purchase_date",
        "order_value_sum": "total_spend",
        "order_value_mean": "average_order_value",
        "order_value_min": "minimum_order_value",
        "order_value_max": "maximum_order_value"
    }
)


# Unique products/categories/sellers at customer level

if product_col:

    product_counts = (
        df.groupby(customer_id_col)[product_col]
        .nunique()
        .rename("unique_products")
    )

    customer_df = customer_df.merge(
        product_counts,
        on=customer_id_col,
        how="left"
    )


if category_col:

    category_counts = (
        df.groupby(customer_id_col)[category_col]
        .nunique()
        .rename("unique_categories")
    )

    customer_df = customer_df.merge(
        category_counts,
        on=customer_id_col,
        how="left"
    )


if seller_col:

    seller_counts = (
        df.groupby(customer_id_col)[seller_col]
        .nunique()
        .rename("unique_sellers")
    )

    customer_df = customer_df.merge(
        seller_counts,
        on=customer_id_col,
        how="left"
    )


# Customer lifetime

customer_df["customer_lifetime_days"] = (
    customer_df["last_purchase_date"]
    - customer_df["first_purchase_date"]
).dt.days


# Days since last purchase relative to dataset end

analysis_end_date = order_df[order_date_col].max()

customer_df["days_since_last_purchase"] = (
    analysis_end_date
    - customer_df["last_purchase_date"]
).dt.days


# Purchase frequency

customer_df["purchase_frequency"] = np.where(
    customer_df["customer_lifetime_days"] > 0,
    customer_df["total_orders"]
    / customer_df["customer_lifetime_days"],
    customer_df["total_orders"]
)


# Repeat customer flag

customer_df["repeat_customer"] = (
    customer_df["total_orders"] >= 2
).astype(int)


customer_df["customer_type"] = np.where(
    customer_df["repeat_customer"] == 1,
    "Repeat Customer",
    "One-Time Customer"
)


logger.info(
    f"Customers: {len(customer_df):,}"
)

logger.info(
    f"Repeat customers: "
    f"{customer_df['repeat_customer'].sum():,}"
)

logger.info(
    f"One-time customers: "
    f"{(customer_df['repeat_customer'] == 0).sum():,}"
)


save_csv(
    customer_df,
    "03_customer_level_features.csv"
)


# ============================================================
# 17. DESCRIPTIVE STATISTICS
# ============================================================

section("STEP 9 — CUSTOMER DESCRIPTIVE STATISTICS")

customer_numeric = customer_df.select_dtypes(
    include=np.number
)

descriptive_stats = (
    customer_numeric
    .describe()
    .T
    .reset_index()
    .rename(columns={"index": "feature"})
)

save_csv(
    descriptive_stats,
    "04_customer_descriptive_statistics.csv"
)


# ============================================================
# 18. REPEAT VS ONE-TIME ANALYSIS
# ============================================================

section("STEP 10 — REPEAT VS ONE-TIME CUSTOMERS")

repeat_summary = (
    customer_df
    .groupby("customer_type")
    .agg(
        customers=(customer_id_col, "nunique"),
        average_orders=("total_orders", "mean"),
        average_spend=("total_spend", "mean"),
        median_spend=("total_spend", "median"),
        average_order_value=(
            "average_order_value",
            "mean"
        ),
        average_lifetime_days=(
            "customer_lifetime_days",
            "mean"
        )
    )
    .reset_index()
)

repeat_summary["percentage"] = (
    repeat_summary["customers"]
    / repeat_summary["customers"].sum()
    * 100
)

save_csv(
    repeat_summary,
    "05_repeat_vs_one_time_customers.csv"
)


# ============================================================
# 19. REPEAT VS ONE-TIME PLOT
# ============================================================

plt.figure()

sns.barplot(
    data=repeat_summary,
    x="customer_type",
    y="percentage"
)

plt.title(
    "Repeat vs One-Time Customers"
)

plt.xlabel("Customer Type")
plt.ylabel("Customers (%)")

save_figure(
    "01_repeat_vs_one_time_customers.png"
)


# ============================================================
# 20. COHORT IDENTIFICATION
# ============================================================

section("STEP 11 — COHORT IDENTIFICATION")

customer_df["cohort_month"] = (
    customer_df["first_purchase_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ============================================================
# 21. ADD COHORT INFO TO ORDER DATA
# ============================================================

order_df = order_df.merge(
    customer_df[
        [
            customer_id_col,
            "cohort_month"
        ]
    ],
    on=customer_id_col,
    how="left"
)

order_df["order_month"] = (
    order_df[order_date_col]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ============================================================
# 22. COHORT PERIOD INDEX
# ============================================================

order_df["cohort_period"] = (
    (order_df["order_month"].dt.year
     - order_df["cohort_month"].dt.year) * 12
    +
    (order_df["order_month"].dt.month
     - order_df["cohort_month"].dt.month)
)


# ============================================================
# 23. RETENTION COUNTS
# ============================================================

section("STEP 12 — COHORT RETENTION ANALYSIS")

cohort_counts = (
    order_df
    .groupby(
        [
            "cohort_month",
            "cohort_period"
        ]
    )[customer_id_col]
    .nunique()
    .reset_index(
        name="retained_customers"
    )
)


# ============================================================
# 24. COHORT SIZE
# ============================================================

cohort_sizes = (
    customer_df
    .groupby("cohort_month")
    [customer_id_col]
    .nunique()
    .reset_index(
        name="cohort_size"
    )
)


# ============================================================
# 25. RETENTION MATRIX
# ============================================================

retention_df = cohort_counts.merge(
    cohort_sizes,
    on="cohort_month",
    how="left"
)

retention_df["retention_rate"] = (
    retention_df["retained_customers"]
    / retention_df["cohort_size"]
    * 100
)

retention_matrix = (
    retention_df
    .pivot(
        index="cohort_month",
        columns="cohort_period",
        values="retention_rate"
    )
)

retention_matrix.columns = [
    f"M{int(col)}"
    for col in retention_matrix.columns
]

retention_matrix = retention_matrix.reset_index()

save_csv(
    retention_matrix,
    "06_cohort_retention_matrix.csv"
)


# ============================================================
# 26. COHORT SUMMARY
# ============================================================

cohort_summary = (
    customer_df
    .groupby("cohort_month")
    .agg(
        cohort_size=(
            customer_id_col,
            "nunique"
        ),
        average_orders=(
            "total_orders",
            "mean"
        ),
        average_spend=(
            "total_spend",
            "mean"
        ),
        repeat_customer_rate=(
            "repeat_customer",
            "mean"
        )
    )
    .reset_index()
)

cohort_summary["repeat_customer_rate"] *= 100

save_csv(
    cohort_summary,
    "07_cohort_summary.csv"
)


# ============================================================
# 27. COHORT RETENTION HEATMAP
# ============================================================

plt.figure(figsize=(16, 10))

heatmap_data = retention_matrix.set_index(
    "cohort_month"
)

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={
        "label": "Retention (%)"
    }
)

plt.title(
    "Customer Cohort Retention Heatmap"
)

plt.xlabel("Months Since First Purchase")

plt.ylabel("Cohort Month")

save_figure(
    "02_cohort_retention_heatmap.png"
)


# ============================================================
# 28. RETENTION CURVES
# ============================================================

plt.figure(figsize=(14, 8))

for cohort in retention_df["cohort_month"].sort_values().unique():

    temp = retention_df[
        retention_df["cohort_month"] == cohort
    ]

    plt.plot(
        temp["cohort_period"],
        temp["retention_rate"],
        marker="o",
        alpha=0.6,
        linewidth=1.5,
        label=str(cohort.date())
    )

plt.title(
    "Cohort Retention Curves"
)

plt.xlabel(
    "Months Since First Purchase"
)

plt.ylabel(
    "Retention Rate (%)"
)

# Avoid unreadable legends for huge cohort counts
if retention_df["cohort_month"].nunique() <= 20:
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8
    )

save_figure(
    "03_cohort_retention_curves.png"
)


# ============================================================
# 29. COHORT SIZE VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 7))

sns.barplot(
    data=cohort_sizes,
    x="cohort_month",
    y="cohort_size"
)

plt.title(
    "Customer Cohort Sizes"
)

plt.xlabel("Cohort Month")
plt.ylabel("Customers")

plt.xticks(rotation=45)

save_figure(
    "04_cohort_sizes.png"
)


# ============================================================
# 30. PURCHASE FREQUENCY
# ============================================================

plt.figure(figsize=(12, 7))

sns.histplot(
    customer_df["total_orders"],
    bins=30,
    kde=False
)

plt.title(
    "Customer Purchase Frequency"
)

plt.xlabel("Number of Orders")

plt.ylabel("Customers")

save_figure(
    "05_purchase_frequency.png"
)


# ============================================================
# 31. CUSTOMER SPENDING
# ============================================================

plt.figure(figsize=(12, 7))

sns.histplot(
    customer_df["total_spend"],
    bins=50,
    kde=False
)

plt.title(
    "Customer Total Spending Distribution"
)

plt.xlabel("Total Spend")

plt.ylabel("Customers")

save_figure(
    "06_customer_spending.png"
)


# ============================================================
# 32. CUSTOMER LIFETIME
# ============================================================

plt.figure(figsize=(12, 7))

sns.histplot(
    customer_df["customer_lifetime_days"],
    bins=50,
    kde=False
)

plt.title(
    "Customer Lifetime Distribution"
)

plt.xlabel("Customer Lifetime (Days)")

plt.ylabel("Customers")

save_figure(
    "07_customer_lifetime.png"
)


# ============================================================
# 33. RFM ANALYSIS
# ============================================================

section("STEP 13 — RFM ANALYSIS")

rfm_df = customer_df[
    [
        customer_id_col,
        "days_since_last_purchase",
        "total_orders",
        "total_spend"
    ]
].copy()

rfm_df = rfm_df.rename(
    columns={
        "days_since_last_purchase": "recency",
        "total_orders": "frequency",
        "total_spend": "monetary"
    }
)


# ============================================================
# 34. RFM SCORES
# ============================================================

# Rank-based scoring is more robust than fixed monetary thresholds.

rfm_df["r_score"] = pd.qcut(
    rfm_df["recency"].rank(method="first"),
    q=5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm_df["f_score"] = pd.qcut(
    rfm_df["frequency"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm_df["m_score"] = pd.qcut(
    rfm_df["monetary"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)


rfm_df["rfm_score"] = (
    rfm_df["r_score"].astype(str)
    + rfm_df["f_score"].astype(str)
    + rfm_df["m_score"].astype(str)
)


# ============================================================
# 35. RFM SEGMENTS
# ============================================================

def assign_rfm_segment(row):

    r = row["r_score"]
    f = row["f_score"]
    m = row["m_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "High-Value Loyal"

    if r >= 4 and f >= 3:
        return "Active Repeat"

    if r <= 2 and f >= 3:
        return "At-Risk Repeat"

    if f <= 2 and m >= 3:
        return "High-Value Occasional"

    if f == 1:
        return "One-Time / New"

    return "Regular"


rfm_df["rfm_segment"] = rfm_df.apply(
    assign_rfm_segment,
    axis=1
)


save_csv(
    rfm_df,
    "08_rfm_customer_segments.csv"
)


# ============================================================
# 36. RFM SUMMARY
# ============================================================

rfm_summary = (
    rfm_df
    .groupby("rfm_segment")
    .agg(
        customers=(customer_id_col, "nunique"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean")
    )
    .reset_index()
)

rfm_summary["percentage"] = (
    rfm_summary["customers"]
    / rfm_summary["customers"].sum()
    * 100
)

save_csv(
    rfm_summary,
    "09_rfm_summary.csv"
)


# ============================================================
# 37. RFM SEGMENT PLOT
# ============================================================

plt.figure(figsize=(13, 7))

segment_plot = (
    rfm_summary
    .sort_values("customers", ascending=False)
)

sns.barplot(
    data=segment_plot,
    x="rfm_segment",
    y="customers"
)

plt.title(
    "Customer Segmentation Based on RFM"
)

plt.xlabel("RFM Segment")

plt.ylabel("Customers")

plt.xticks(rotation=35)

save_figure(
    "08_rfm_customer_segments.png"
)


# ============================================================
# 38. OUTLIER ANALYSIS
# ============================================================

section("STEP 14 — OUTLIER ANALYSIS")

outlier_features = [
    "total_orders",
    "total_spend",
    "average_order_value",
    "customer_lifetime_days",
    "days_since_last_purchase"
]

outlier_features = [
    col
    for col in outlier_features
    if col in customer_df.columns
]


outlier_results = []

for col in outlier_features:

    series = customer_df[col].dropna()

    if len(series) == 0:
        continue

    lower, upper = iqr_bounds(series)

    mask = (
        (customer_df[col] < lower)
        | (customer_df[col] > upper)
    )

    outlier_count = mask.sum()

    outlier_results.append({
        "feature": col,
        "q1": series.quantile(0.25),
        "q3": series.quantile(0.75),
        "iqr": series.quantile(0.75)
        - series.quantile(0.25),
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": outlier_count,
        "outlier_percentage": (
            outlier_count
            / len(customer_df)
            * 100
        )
    })


outlier_summary = pd.DataFrame(
    outlier_results
)

save_csv(
    outlier_summary,
    "10_outlier_summary.csv"
)


# ============================================================
# 39. OUTLIER BOXPLOTS
# ============================================================

for col in outlier_features:

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        x=customer_df[col]
    )

    plt.title(
        f"Outlier Analysis — {col}"
    )

    plt.xlabel(col)

    filename = (
        f"outlier_{col}.png"
    )

    save_figure(filename)


# ============================================================
# 40. CORRELATION ANALYSIS
# ============================================================

section("STEP 15 — CORRELATION ANALYSIS")

correlation_features = [
    "total_orders",
    "total_spend",
    "average_order_value",
    "customer_lifetime_days",
    "days_since_last_purchase",
    "purchase_frequency"
]

correlation_features = [
    col
    for col in correlation_features
    if col in customer_df.columns
]

corr_matrix = customer_df[
    correlation_features
].corr(
    method="spearman"
)

corr_matrix.to_csv(
    TABLE_DIR / "11_spearman_correlation_matrix.csv"
)


plt.figure(figsize=(11, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Customer Behavioral Feature Correlation"
)

save_figure(
    "09_customer_correlation.png"
)


# ============================================================
# 41. RETENTION BEHAVIOR COMPARISON
# ============================================================

section("STEP 16 — RETAINED VS NON-RETAINED BEHAVIOR")

# For Part 1, retained is defined as:
#
# Customer has made at least one additional order
# after the first order.
#
# This is historical repeat-purchase retention.
#
# This definition is intentionally kept separate from
# Part 2's future prediction target.

customer_behavior = (
    customer_df
    .groupby("customer_type")
    .agg(
        customers=(customer_id_col, "nunique"),
        avg_orders=("total_orders", "mean"),
        median_orders=("total_orders", "median"),
        avg_spend=("total_spend", "mean"),
        median_spend=("total_spend", "median"),
        avg_aov=("average_order_value", "mean"),
        avg_lifetime=(
            "customer_lifetime_days",
            "mean"
        ),
        avg_recency=(
            "days_since_last_purchase",
            "mean"
        )
    )
    .reset_index()
)

save_csv(
    customer_behavior,
    "12_retained_vs_non_retained_behavior.csv"
)


# ============================================================
# 42. STATISTICAL TESTS
# ============================================================

section("STEP 17 — STATISTICAL ANALYSIS")

statistical_results = []


# ------------------------------------------------------------
# Mann-Whitney U Tests
# ------------------------------------------------------------

retained = customer_df[
    customer_df["repeat_customer"] == 1
]

non_retained = customer_df[
    customer_df["repeat_customer"] == 0
]


for feature in [
    "total_spend",
    "average_order_value",
    "customer_lifetime_days",
    "days_since_last_purchase"
]:

    if feature not in customer_df.columns:
        continue

    a = retained[feature].dropna()
    b = non_retained[feature].dropna()

    if len(a) > 0 and len(b) > 0:

        try:

            statistic, p_value = stats.mannwhitneyu(
                a,
                b,
                alternative="two-sided"
            )

            statistical_results.append({
                "test": "Mann-Whitney U",
                "feature": feature,
                "group_1": "Repeat Customer",
                "group_2": "One-Time Customer",
                "statistic": statistic,
                "p_value": p_value
            })

        except Exception:
            pass


# ============================================================
# 43. CHI-SQUARE: RFM SEGMENT vs CUSTOMER TYPE
# ============================================================

contingency = pd.crosstab(
    rfm_df["rfm_segment"],
    customer_df.set_index(
        customer_id_col
    )["customer_type"]
)


if contingency.shape[0] > 1 and contingency.shape[1] > 1:

    chi2, p, dof, expected = (
        stats.chi2_contingency(
            contingency
        )
    )

    cv = cramers_v(contingency)

    statistical_results.append({
        "test": "Chi-Square",
        "feature": "RFM Segment vs Customer Type",
        "group_1": "RFM Segment",
        "group_2": "Customer Type",
        "statistic": chi2,
        "p_value": p,
        "cramers_v": cv
    })


statistical_results = pd.DataFrame(
    statistical_results
)

save_csv(
    statistical_results,
    "13_statistical_tests.csv"
)


# ============================================================
# 44. MULTIPLE TESTING CORRECTION
# ============================================================

if (
    not statistical_results.empty
    and "p_value" in statistical_results.columns
):

    p_values = (
        statistical_results["p_value"]
        .fillna(1)
        .values
    )

    m = len(p_values)

    order = np.argsort(p_values)

    adjusted = np.empty(m)

    for rank, index in enumerate(order, start=1):

        adjusted[index] = (
            p_values[index]
            * m
            / rank
        )

    adjusted = np.minimum(
        adjusted,
        1
    )

    statistical_results[
        "bh_adjusted_p_value"
    ] = adjusted

    statistical_results[
        "significant_after_bh_0_05"
    ] = (
        statistical_results[
            "bh_adjusted_p_value"
        ] < 0.05
    )

    save_csv(
        statistical_results,
        "14_statistical_tests_bh_corrected.csv"
    )


# ============================================================
# 45. OPTIONAL K-MEANS CUSTOMER SEGMENTATION
# ============================================================

section("STEP 18 — BEHAVIORAL CUSTOMER CLUSTERING")

cluster_features = [
    "recency",
    "frequency",
    "monetary"
]

cluster_data = rfm_df[
    cluster_features
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()


cluster_result = None


if len(cluster_data) >= 100:

    # Sample for faster cluster selection on very large datasets

    if len(cluster_data) > 50000:

        cluster_sample = (
            cluster_data
            .sample(
                50000,
                random_state=RANDOM_STATE
            )
        )

    else:

        cluster_sample = cluster_data


    scaler = StandardScaler()

    scaled_sample = scaler.fit_transform(
        cluster_sample
    )


    silhouette_results = []

    for k in range(2, 7):

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        labels = model.fit_predict(
            scaled_sample
        )

        score = silhouette_score(
            scaled_sample,
            labels
        )

        silhouette_results.append({
            "k": k,
            "silhouette_score": score
        })


    silhouette_df = pd.DataFrame(
        silhouette_results
    )

    save_csv(
        silhouette_df,
        "15_clustering_silhouette_scores.csv"
    )


    best_k = int(
        silhouette_df.loc[
            silhouette_df[
                "silhouette_score"
            ].idxmax(),
            "k"
        ]
    )


    logger.info(
        f"Selected clustering k={best_k}"
    )


    # Fit final model on all customer RFM data

    full_cluster_data = (
        rfm_df[
            cluster_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(
            rfm_df[
                cluster_features
            ].median()
        )
    )


    scaler_final = StandardScaler()

    scaled_full = scaler_final.fit_transform(
        full_cluster_data
    )


    final_kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10
    )


    rfm_df["behavior_cluster"] = (
        final_kmeans.fit_predict(
            scaled_full
        )
    )


    cluster_profile = (
        rfm_df
        .groupby("behavior_cluster")
        .agg(
            customers=(customer_id_col, "nunique"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean")
        )
        .reset_index()
    )


    cluster_profile["percentage"] = (
        cluster_profile["customers"]
        / cluster_profile["customers"].sum()
        * 100
    )


    save_csv(
        cluster_profile,
        "16_behavior_cluster_profile.csv"
    )


    # Attach cluster to customer dataframe

    customer_df = customer_df.merge(
        rfm_df[
            [
                customer_id_col,
                "behavior_cluster"
            ]
        ],
        on=customer_id_col,
        how="left"
    )


    # Cluster visualization

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=rfm_df,
        x="frequency",
        y="monetary",
        hue="behavior_cluster",
        palette="tab10",
        alpha=0.6
    )

    plt.title(
        "Customer Behavioral Clusters"
    )

    plt.xlabel("Purchase Frequency")

    plt.ylabel("Monetary Value")

    save_figure(
        "10_customer_behavior_clusters.png"
    )


else:

    logger.warning(
        "Not enough customer records for clustering."
    )


# ============================================================
# 46. RETENTION BY RFM SEGMENT
# ============================================================

section("STEP 19 — RETENTION BY CUSTOMER SEGMENT")

retention_by_segment = (
    rfm_df
    .merge(
        customer_df[
            [
                customer_id_col,
                "repeat_customer"
            ]
        ],
        on=customer_id_col,
        how="left"
    )
    .groupby("rfm_segment")
    .agg(
        customers=(
            customer_id_col,
            "nunique"
        ),
        retention_rate=(
            "repeat_customer",
            "mean"
        ),
        avg_recency=(
            "recency",
            "mean"
        ),
        avg_frequency=(
            "frequency",
            "mean"
        ),
        avg_monetary=(
            "monetary",
            "mean"
        )
    )
    .reset_index()
)

retention_by_segment[
    "retention_rate"
] *= 100

save_csv(
    retention_by_segment,
    "17_retention_by_rfm_segment.csv"
)


# ============================================================
# 47. COHORT MONTH-1 RETENTION
# ============================================================

section("STEP 20 — COHORT PERFORMANCE")

month_1 = retention_df[
    retention_df["cohort_period"] == 1
].copy()

if not month_1.empty:

    month_1 = month_1[
        [
            "cohort_month",
            "cohort_size",
            "retained_customers",
            "retention_rate"
        ]
    ]

    month_1 = month_1.sort_values(
        "retention_rate",
        ascending=False
    )

    save_csv(
        month_1,
        "18_month_1_cohort_performance.csv"
    )


# ============================================================
# 48. NORMALITY / DISTRIBUTION SUMMARY
# ============================================================

section("STEP 21 — DISTRIBUTION ANALYSIS")

distribution_results = []

for feature in [
    "total_orders",
    "total_spend",
    "average_order_value",
    "customer_lifetime_days",
    "days_since_last_purchase"
]:

    if feature not in customer_df.columns:
        continue

    series = (
        customer_df[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(series) == 0:
        continue

    # For very large datasets, use a deterministic sample
    # for Shapiro-Wilk because it is not appropriate to run
    # on the full 100k+ dataset.

    sample_size = min(
        5000,
        len(series)
    )

    sample = series.sample(
        sample_size,
        random_state=RANDOM_STATE
    )

    try:

        shapiro_stat, shapiro_p = (
            stats.shapiro(sample)
        )

    except Exception:

        shapiro_stat = np.nan
        shapiro_p = np.nan


    try:

        normality_stat, normality_p = (
            stats.normaltest(sample)
        )

    except Exception:

        normality_stat = np.nan
        normality_p = np.nan


    distribution_results.append({
        "feature": feature,
        "skewness": stats.skew(series),
        "kurtosis": stats.kurtosis(series),
        "sample_size_for_shapiro": sample_size,
        "shapiro_statistic": shapiro_stat,
        "shapiro_p_value": shapiro_p,
        "dagostino_statistic": normality_stat,
        "dagostino_p_value": normality_p
    })


distribution_df = pd.DataFrame(
    distribution_results
)

save_csv(
    distribution_df,
    "19_distribution_normality_analysis.csv"
)


# ============================================================
# 49. SKEWNESS SUMMARY
# ============================================================

skewness_df = distribution_df[
    [
        "feature",
        "skewness",
        "kurtosis"
    ]
].copy()

skewness_df["skewness_interpretation"] = np.select(
    [
        skewness_df["skewness"] < -1,
        skewness_df["skewness"] <= 1,
        skewness_df["skewness"] > 1
    ],
    [
        "Strong negative skew",
        "Approximately symmetric/moderate skew",
        "Strong positive skew"
    ],
    default="Unknown"
)

save_csv(
    skewness_df,
    "20_skewness_kurtosis_analysis.csv"
)


# ============================================================
# 50. FINAL CUSTOMER DATASET
# ============================================================

section("STEP 22 — FINAL CUSTOMER DATASET")

# Merge RFM features into final customer-level dataset

rfm_columns_to_merge = [
    customer_id_col,
    "recency",
    "frequency",
    "monetary",
    "r_score",
    "f_score",
    "m_score",
    "rfm_score",
    "rfm_segment"
]

if "behavior_cluster" in rfm_df.columns:
    rfm_columns_to_merge.append(
        "behavior_cluster"
    )


final_customer_df = customer_df.merge(
    rfm_df[rfm_columns_to_merge],
    on=customer_id_col,
    how="left"
)


save_csv(
    final_customer_df,
    "21_final_customer_analytics_dataset.csv"
)


# ============================================================
# 51. OVERALL PROJECT METRICS
# ============================================================

section("STEP 23 — FINAL PROJECT METRICS")

total_customers = (
    final_customer_df[
        customer_id_col
    ].nunique()
)

repeat_customers = (
    final_customer_df[
        "repeat_customer"
    ].sum()
)

one_time_customers = (
    total_customers
    - repeat_customers
)

repeat_rate = (
    repeat_customers
    / total_customers
    * 100
    if total_customers > 0
    else np.nan
)

one_time_rate = (
    one_time_customers
    / total_customers
    * 100
    if total_customers > 0
    else np.nan
)

total_orders = (
    order_df[
        order_id_col
    ].nunique()
)

total_spend = (
    order_df["order_value"]
    .sum()
)


project_metrics = pd.DataFrame({
    "metric": [
        "Initial rows",
        "Initial columns",
        "Final transaction rows",
        "Unique customers",
        "Unique orders",
        "Repeat customers",
        "One-time customers",
        "Repeat customer percentage",
        "One-time customer percentage",
        "Total spend",
        "Analysis start date",
        "Analysis end date",
        "Number of cohorts"
    ],
    "value": [
        initial_rows,
        initial_columns,
        len(df),
        total_customers,
        total_orders,
        repeat_customers,
        one_time_customers,
        repeat_rate,
        one_time_rate,
        total_spend,
        order_df[order_date_col].min(),
        order_df[order_date_col].max(),
        customer_df["cohort_month"].nunique()
    ]
})

save_csv(
    project_metrics,
    "22_final_project_metrics.csv"
)


# ============================================================
# 52. BUSINESS INSIGHTS
# ============================================================

section("STEP 24 — GENERATING BUSINESS INSIGHTS")

insights = []

insights.append(
    f"Total unique customers analyzed: "
    f"{total_customers:,}."
)

insights.append(
    f"Total unique orders analyzed: "
    f"{total_orders:,}."
)

insights.append(
    f"Repeat customers: "
    f"{repeat_customers:,} "
    f"({repeat_rate:.2f}%)."
)

insights.append(
    f"One-time customers: "
    f"{one_time_customers:,} "
    f"({one_time_rate:.2f}%)."
)


if not cohort_summary.empty:

    largest_cohort = cohort_summary.loc[
        cohort_summary["cohort_size"].idxmax()
    ]

    insights.append(
        f"Largest acquisition cohort: "
        f"{largest_cohort['cohort_month'].date()} "
        f"with "
        f"{int(largest_cohort['cohort_size']):,} customers."
    )


if not month_1.empty:

    valid_month_1 = month_1.dropna(
        subset=["retention_rate"]
    )

    if not valid_month_1.empty:

        best_cohort = valid_month_1.iloc[0]

        worst_cohort = valid_month_1.iloc[-1]

        insights.append(
            f"Highest observed Month-1 retention cohort: "
            f"{best_cohort['cohort_month'].date()} "
            f"at "
            f"{best_cohort['retention_rate']:.2f}%."
        )

        insights.append(
            f"Lowest observed Month-1 retention cohort: "
            f"{worst_cohort['cohort_month'].date()} "
            f"at "
            f"{worst_cohort['retention_rate']:.2f}%."
        )


if not rfm_summary.empty:

    largest_segment = rfm_summary.loc[
        rfm_summary["customers"].idxmax()
    ]

    insights.append(
        f"Largest RFM segment: "
        f"{largest_segment['rfm_segment']} "
        f"({int(largest_segment['customers']):,} customers)."
    )


if not outlier_summary.empty:

    highest_outlier = outlier_summary.loc[
        outlier_summary[
            "outlier_percentage"
        ].idxmax()
    ]

    insights.append(
        f"Highest IQR outlier proportion was observed in "
        f"{highest_outlier['feature']} "
        f"({highest_outlier['outlier_percentage']:.2f}%). "
        f"Outliers were retained rather than blindly removed."
    )


# ============================================================
# 53. FINAL REPORT
# ============================================================

report_path = (
    REPORT_DIR
    / "part1_final_summary.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CUSTOMER COHORT RETENTION ANALYSIS\n"
    )

    f.write(
        "PART 1 — FINAL ANALYTICS REPORT\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        "PROJECT OBJECTIVE\n"
    )

    f.write(
        "Analyze customer acquisition cohorts, repeat-purchase "
        "behavior, retention patterns, RFM characteristics, "
        "customer segments, and business-level retention insights.\n\n"
    )

    f.write(
        "RETENTION DEFINITION\n"
    )

    f.write(
        "For Part 1, a customer is considered a repeat/retained "
        "customer when the customer has at least one additional "
        "order beyond the first observed order. Cohort retention "
        "is calculated by tracking unique customers purchasing "
        "during each month relative to their cohort month.\n\n"
    )

    f.write(
        "DATASET SUMMARY\n"
    )

    for _, row in project_metrics.iterrows():

        f.write(
            f"{row['metric']}: {row['value']}\n"
        )

    f.write("\n")

    f.write(
        "BUSINESS INSIGHTS\n"
    )

    for i, insight in enumerate(
        insights,
        start=1
    ):

        f.write(
            f"{i}. {insight}\n"
        )

    f.write("\n")

    f.write(
        "METHODOLOGY\n"
    )

    f.write(
        "1. Data understanding and quality assessment\n"
        "2. Duplicate and invalid-value handling\n"
        "3. Date processing\n"
        "4. Customer-order aggregation\n"
        "5. Customer-level feature engineering\n"
        "6. RFM analysis\n"
        "7. Cohort identification\n"
        "8. Cohort retention calculation\n"
        "9. Retention visualization\n"
        "10. Repeat vs one-time customer analysis\n"
        "11. Outlier analysis\n"
        "12. Correlation analysis\n"
        "13. Statistical testing\n"
        "14. Customer segmentation\n"
        "15. Business insight generation\n\n"
    )

    f.write(
        "PART 2 NOTE\n"
    )

    f.write(
        "Part 2 will use the analytical foundation created here "
        "for machine-learning-based future retention prediction. "
        "Prediction-window definition, target creation, "
        "time-aware validation, leakage prevention, model training, "
        "hyperparameter optimization, calibration, and SHAP "
        "explainability should be implemented separately.\n"
    )


logger.info(
    f"Final report saved: {report_path}"
)


# ============================================================
# 54. FINAL SUMMARY PRINT
# ============================================================

section("PART 1 COMPLETED")

logger.info(
    f"Customers analyzed: {total_customers:,}"
)

logger.info(
    f"Orders analyzed: {total_orders:,}"
)

logger.info(
    f"Repeat customers: "
    f"{repeat_customers:,} ({repeat_rate:.2f}%)"
)

logger.info(
    f"One-time customers: "
    f"{one_time_customers:,} ({one_time_rate:.2f}%)"
)

logger.info(
    f"Cohorts analyzed: "
    f"{customer_df['cohort_month'].nunique():,}"
)

logger.info(
    f"Output directory: {OUTPUT_ROOT}"
)

logger.info(
    "PART 1 ANALYTICS PIPELINE FINISHED SUCCESSFULLY."
)

logger.info(
    "PART 2 ML/PREDICTION HAS NOT BEEN INCLUDED."
)




# ============================================================
# CUSTOMER COHORT RETENTION ANALYSIS
# PART 2 — CUSTOMER RETENTION PREDICTION
# ============================================================
#
# GOAL:
# Predict whether a customer will make a repeat purchase
# during the future prediction window.
#
# RETENTION DEFINITION:
# Customer = Retained (1)
# if they make at least one purchase during the
# FUTURE_WINDOW_DAYS after the observation window.
#
# Designed for:
# - Fast execution
# - Leakage prevention
# - Time-aware validation
# - Optuna optimization
# - GitHub-friendly outputs
#
# YOU ONLY NEED TO CHANGE:
# DATA_PATH
#
# ============================================================

from pathlib import Path
import warnings
import json
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

import optuna
import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

# ============================================================
# CHANGE ONLY THIS PATH
# ============================================================

DATA_PATH = Path(
    r"C:Customer.py/olist_customer_cohort_merged.csv"
)

# ============================================================

RANDOM_STATE = 42

# Observation period:
# Customer history used to create predictive features.
OBSERVATION_WINDOW_DAYS = 90

# Future period used to define retention.
FUTURE_WINDOW_DAYS = 90

# Minimum purchase required during observation window.
# A customer needs at least one initial purchase.
MIN_OBSERVATION_ORDERS = 1

# Percentage of chronological snapshots used for final test.
TEST_SIZE = 0.20

# Optuna
N_TRIALS = 100

# Maximum rows used during tuning for speed.
# Final model is trained using all training data.
TUNING_MAX_ROWS = 40000

# Number of CV folds.
N_SPLITS = 2

OUTPUT_DIR = Path("outputs") / "part2"

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

for directory in [
    TABLE_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Find the first matching column from candidate names.
    Matching is case-insensitive.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def calculate_metrics(y_true, probabilities, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
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
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),
        "Brier_Score": brier_score_loss(
            y_true,
            probabilities
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# 3. LOAD DATA
# ============================================================

start_time = time.time()

print_section("LOADING DATA")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Change DATA_PATH at the top of the script."
    )

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


# ============================================================
# 4. AUTOMATICALLY IDENTIFY IMPORTANT COLUMNS
# ============================================================

print_section("IDENTIFYING IMPORTANT COLUMNS")

CUSTOMER_COL = find_column(
    df,
    [
        "customer_unique_id",
        "customer_id",
        "CustomerID",
        "customer"
    ]
)

ORDER_COL = find_column(
    df,
    [
        "order_id",
        "OrderID"
    ]
)

DATE_COL = find_column(
    df,
    [
        "order_purchase_timestamp",
        "purchase_timestamp",
        "order_date",
        "purchase_date",
        "OrderDate"
    ]
)

PRICE_COL = find_column(
    df,
    [
        "price",
        "payment_value",
        "order_value",
        "total_price"
    ]
)

PRODUCT_COL = find_column(
    df,
    [
        "product_id",
        "ProductID"
    ]
)

CATEGORY_COL = find_column(
    df,
    [
        "product_category_name",
        "category",
        "product_category"
    ]
)

SELLER_COL = find_column(
    df,
    [
        "seller_id",
        "SellerID"
    ]
)

print(f"Customer column : {CUSTOMER_COL}")
print(f"Order column    : {ORDER_COL}")
print(f"Date column     : {DATE_COL}")
print(f"Price column    : {PRICE_COL}")
print(f"Product column  : {PRODUCT_COL}")
print(f"Category column : {CATEGORY_COL}")
print(f"Seller column   : {SELLER_COL}")


if CUSTOMER_COL is None:
    raise ValueError(
        "Customer ID column could not be identified."
    )

if ORDER_COL is None:
    raise ValueError(
        "Order ID column could not be identified."
    )

if DATE_COL is None:
    raise ValueError(
        "Purchase/order date column could not be identified."
    )


# ============================================================
# 5. PREPARE CORE COLUMNS
# ============================================================

print_section("PREPARING CORE DATA")

keep_cols = [
    CUSTOMER_COL,
    ORDER_COL,
    DATE_COL,
]

for col in [
    PRICE_COL,
    PRODUCT_COL,
    CATEGORY_COL,
    SELLER_COL,
]:
    if col is not None and col not in keep_cols:
        keep_cols.append(col)

data = df[keep_cols].copy()

del df

data[DATE_COL] = pd.to_datetime(
    data[DATE_COL],
    errors="coerce"
)

data = data.dropna(
    subset=[
        CUSTOMER_COL,
        ORDER_COL,
        DATE_COL
    ]
)

data[CUSTOMER_COL] = data[CUSTOMER_COL].astype(str)
data[ORDER_COL] = data[ORDER_COL].astype(str)

if PRICE_COL is not None:
    data[PRICE_COL] = safe_numeric(
        data[PRICE_COL]
    ).fillna(0)

data = data.sort_values(
    DATE_COL
)

print(f"Clean transaction rows: {len(data):,}")
print(
    f"Date range: "
    f"{data[DATE_COL].min()} → {data[DATE_COL].max()}"
)


# ============================================================
# 6. CREATE ORDER-LEVEL DATA
# ============================================================
#
# IMPORTANT:
# If merged dataset contains multiple rows per order
# because of products/items, aggregate to one row per order.
#
# This prevents counting one order multiple times.
# ============================================================

print_section("CREATING ORDER-LEVEL DATA")

agg_dict = {}

if PRICE_COL is not None:
    agg_dict[PRICE_COL] = "sum"

if PRODUCT_COL is not None:
    agg_dict[PRODUCT_COL] = "nunique"

if CATEGORY_COL is not None:
    agg_dict[CATEGORY_COL] = "nunique"

if SELLER_COL is not None:
    agg_dict[SELLER_COL] = "nunique"

order_data = (
    data.groupby(
        [
            CUSTOMER_COL,
            ORDER_COL
        ],
        as_index=False
    )
    .agg(
        {
            DATE_COL: "min",
            **agg_dict
        }
    )
)

if PRICE_COL is None:
    order_data["order_value"] = 0
    PRICE_FEATURE = "order_value"
else:
    PRICE_FEATURE = PRICE_COL

print(f"Unique customer-order records: {len(order_data):,}")


# ============================================================
# 7. CREATE CUSTOMER SNAPSHOT DATASET
# ============================================================
#
# Design:
#
# First purchase
#      ↓
# Observation window = 90 days
#      ↓
# Features created only from observation window
#      ↓
# Future 90 days
#      ↓
# Retention target
#
# This prevents future information from entering features.
# ============================================================

print_section("CREATING LEAKAGE-FREE CUSTOMER SNAPSHOT")

global_max_date = order_data[DATE_COL].max()

customer_first = (
    order_data
    .groupby(CUSTOMER_COL)[DATE_COL]
    .min()
    .rename("first_purchase_date")
)

order_data = order_data.merge(
    customer_first,
    on=CUSTOMER_COL,
    how="left"
)

order_data["days_from_first"] = (
    order_data[DATE_COL]
    - order_data["first_purchase_date"]
).dt.days


# ============================================================
# 8. CUSTOMER FEATURES FROM OBSERVATION WINDOW
# ============================================================

observation_data = order_data[
    order_data["days_from_first"]
    <= OBSERVATION_WINDOW_DAYS
].copy()

print(
    f"Observation-window orders: "
    f"{len(observation_data):,}"
)


customer_features = (
    observation_data
    .groupby(CUSTOMER_COL)
    .agg(
        observation_orders=(
            ORDER_COL,
            "nunique"
        ),
        total_spend_so_far=(
            PRICE_FEATURE,
            "sum"
        ),
        average_order_value=(
            PRICE_FEATURE,
            "mean"
        ),
        max_order_value=(
            PRICE_FEATURE,
            "max"
        ),
        first_purchase_date=(
            "first_purchase_date",
            "first"
        ),
        last_observed_purchase_date=(
            DATE_COL,
            "max"
        ),
    )
    .reset_index()
)


# ============================================================
# 9. ADD BEHAVIORAL FEATURES
# ============================================================

print_section("FEATURE ENGINEERING")

# Recency
customer_features["recency_days"] = (
    customer_features["last_observed_purchase_date"]
    - customer_features["first_purchase_date"]
).dt.days

# Observation activity duration
customer_features["observation_active_days"] = (
    customer_features["last_observed_purchase_date"]
    - customer_features["first_purchase_date"]
).dt.days

# Frequency
customer_features["purchase_frequency"] = (
    customer_features["observation_orders"]
    /
    (
        customer_features["observation_active_days"]
        .clip(lower=1)
        + 1
    )
)

# Log-transformed spend
customer_features["log_total_spend"] = np.log1p(
    customer_features["total_spend_so_far"].clip(lower=0)
)

# Log order count
customer_features["log_order_count"] = np.log1p(
    customer_features["observation_orders"]
)


# ============================================================
# 10. PRODUCT / CATEGORY / SELLER FEATURES
# ============================================================

if PRODUCT_COL is not None:

    product_counts = (
        observation_data
        .groupby(CUSTOMER_COL)[PRODUCT_COL]
        .nunique()
        .rename("unique_products")
    )

    customer_features = customer_features.merge(
        product_counts,
        on=CUSTOMER_COL,
        how="left"
    )

else:

    customer_features["unique_products"] = 0


if CATEGORY_COL is not None:

    category_counts = (
        observation_data
        .groupby(CUSTOMER_COL)[CATEGORY_COL]
        .nunique()
        .rename("unique_categories")
    )

    customer_features = customer_features.merge(
        category_counts,
        on=CUSTOMER_COL,
        how="left"
    )

else:

    customer_features["unique_categories"] = 0


if SELLER_COL is not None:

    seller_counts = (
        observation_data
        .groupby(CUSTOMER_COL)[SELLER_COL]
        .nunique()
        .rename("unique_sellers")
    )

    customer_features = customer_features.merge(
        seller_counts,
        on=CUSTOMER_COL,
        how="left"
    )

else:

    customer_features["unique_sellers"] = 0


# ============================================================
# 11. REMOVE CUSTOMERS WITHOUT ENOUGH OBSERVATION
# ============================================================

customer_features = customer_features[
    customer_features["observation_orders"]
    >= MIN_OBSERVATION_ORDERS
].copy()


# ============================================================
# 12. CREATE FUTURE RETENTION TARGET
# ============================================================

print_section("CREATING FUTURE RETENTION TARGET")

# Merge customer observation point
customer_features["prediction_point"] = (
    customer_features["first_purchase_date"]
    + pd.to_timedelta(
        OBSERVATION_WINDOW_DAYS,
        unit="D"
    )
)

customer_features["future_window_end"] = (
    customer_features["prediction_point"]
    + pd.to_timedelta(
        FUTURE_WINDOW_DAYS,
        unit="D"
    )
)

# Merge only necessary columns
future_orders = order_data[
    [
        CUSTOMER_COL,
        DATE_COL
    ]
].copy()

future_orders = future_orders.merge(
    customer_features[
        [
            CUSTOMER_COL,
            "prediction_point",
            "future_window_end"
        ]
    ],
    on=CUSTOMER_COL,
    how="inner"
)

future_orders = future_orders[
    (
        future_orders[DATE_COL]
        > future_orders["prediction_point"]
    )
    &
    (
        future_orders[DATE_COL]
        <= future_orders["future_window_end"]
    )
]

future_counts = (
    future_orders
    .groupby(CUSTOMER_COL)
    .size()
    .rename("future_orders")
)

customer_features = customer_features.merge(
    future_counts,
    on=CUSTOMER_COL,
    how="left"
)

customer_features["future_orders"] = (
    customer_features["future_orders"]
    .fillna(0)
    .astype(int)
)

customer_features["retained"] = (
    customer_features["future_orders"] > 0
).astype(int)


# ============================================================
# 13. REMOVE CUSTOMERS WITHOUT COMPLETE FUTURE WINDOW
# ============================================================
#
# If the dataset ends before the complete prediction window,
# we cannot know whether those customers were retained.
#
# Therefore they are excluded rather than incorrectly labeled 0.
# ============================================================

customer_features = customer_features[
    customer_features["future_window_end"]
    <= global_max_date
].copy()


print(
    f"Final prediction dataset: "
    f"{customer_features.shape}"
)


# ============================================================
# 14. SAVE PREDICTION DATASET SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "rows",
            "features_before_modeling",
            "retained_customers",
            "non_retained_customers",
            "retention_rate",
            "observation_window_days",
            "future_window_days",
        ],
        "value": [
            len(customer_features),
            customer_features.shape[1],
            int(customer_features["retained"].sum()),
            int((customer_features["retained"] == 0).sum()),
            customer_features["retained"].mean(),
            OBSERVATION_WINDOW_DAYS,
            FUTURE_WINDOW_DAYS,
        ],
    }
)

summary.to_csv(
    TABLE_DIR / "prediction_dataset_summary.csv",
    index=False
)


# ============================================================
# 15. CLASS DISTRIBUTION
# ============================================================

class_distribution = (
    customer_features["retained"]
    .value_counts()
    .sort_index()
    .rename_axis("retained")
    .reset_index(name="count")
)

class_distribution["percentage"] = (
    class_distribution["count"]
    / class_distribution["count"].sum()
    * 100
)

class_distribution.to_csv(
    TABLE_DIR / "class_distribution.csv",
    index=False
)

print("\nTarget distribution:")
print(class_distribution)


# ============================================================
# 16. TEMPORAL SPLIT
# ============================================================

print_section("TIME-BASED TRAIN / TEST SPLIT")

customer_features = customer_features.sort_values(
    "prediction_point"
).reset_index(drop=True)

split_index = int(
    len(customer_features)
    * (1 - TEST_SIZE)
)

train_full = customer_features.iloc[
    :split_index
].copy()

test = customer_features.iloc[
    split_index:
].copy()

print(f"Training rows: {len(train_full):,}")
print(f"Test rows    : {len(test):,}")

print(
    f"\nTrain period:\n"
    f"{train_full['prediction_point'].min()} → "
    f"{train_full['prediction_point'].max()}"
)

print(
    f"\nTest period:\n"
    f"{test['prediction_point'].min()} → "
    f"{test['prediction_point'].max()}"
)


# ============================================================
# 17. TRAIN / VALIDATION SPLIT
# ============================================================
#
# Last 20% of training period = validation.
# Final test remains untouched.
# ============================================================

validation_index = int(
    len(train_full) * 0.80
)

train = train_full.iloc[
    :validation_index
].copy()

validation = train_full.iloc[
    validation_index:
].copy()

TARGET = "retained"

# ============================================================
# 18. DROP LEAKAGE / NON-MODELING COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = [
    CUSTOMER_COL,
    TARGET,

    # Future information
    "future_orders",
    "prediction_point",
    "future_window_end",

    # Dates are transformed into useful features instead
    "first_purchase_date",
    "last_observed_purchase_date",
]

NON_FEATURE_COLUMNS = [
    col for col in NON_FEATURE_COLUMNS
    if col in customer_features.columns
]


def prepare_X(dataframe):

    X = dataframe.drop(
        columns=NON_FEATURE_COLUMNS,
        errors="ignore"
    ).copy()

    return X


X_train = prepare_X(train)
X_valid = prepare_X(validation)
X_test = prepare_X(test)

y_train = train[TARGET]
y_valid = validation[TARGET]
y_test = test[TARGET]


# ============================================================
# 19. REMOVE SUSPICIOUS ID-LIKE COLUMNS
# ============================================================

print_section("FEATURE QUALITY CHECK")

id_like_columns = []

for col in X_train.columns:

    if X_train[col].dtype == "object":

        unique_ratio = (
            X_train[col].nunique(dropna=True)
            / max(len(X_train), 1)
        )

        if unique_ratio > 0.50:
            id_like_columns.append(col)


if id_like_columns:

    print(
        "Removing high-cardinality ID-like columns:"
    )

    for col in id_like_columns:
        print(f"  - {col}")

    X_train = X_train.drop(
        columns=id_like_columns
    )

    X_valid = X_valid.drop(
        columns=id_like_columns,
        errors="ignore"
    )

    X_test = X_test.drop(
        columns=id_like_columns,
        errors="ignore"
    )


# ============================================================
# 20. REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    col
    for col in X_train.columns
    if X_train[col].nunique(dropna=False) <= 1
]

if constant_columns:

    print(
        "\nRemoving constant features:"
    )

    for col in constant_columns:
        print(f"  - {col}")

    X_train = X_train.drop(
        columns=constant_columns
    )

    X_valid = X_valid.drop(
        columns=constant_columns,
        errors="ignore"
    )

    X_test = X_test.drop(
        columns=constant_columns,
        errors="ignore"
    )


# ============================================================
# 21. FEATURE METADATA
# ============================================================

feature_metadata = pd.DataFrame(
    {
        "feature": X_train.columns,
        "dtype": [
            str(X_train[c].dtype)
            for c in X_train.columns
        ],
        "missing_count": [
            int(X_train[c].isna().sum())
            for c in X_train.columns
        ],
        "unique_values": [
            int(X_train[c].nunique(dropna=True))
            for c in X_train.columns
        ],
    }
)

feature_metadata.to_csv(
    TABLE_DIR / "feature_metadata.csv",
    index=False
)

print(
    f"\nFinal raw feature count: "
    f"{X_train.shape[1]}"
)


# ============================================================
# 22. PREPROCESSING
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print(
    f"Numeric features    : {len(numeric_features)}"
)

print(
    f"Categorical features: {len(categorical_features)}"
)


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
        ),
    ]
)

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
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                sparse_output=False
            )
        ),
    ]
)

transformers = []

if numeric_features:

    transformers.append(
        (
            "num",
            numeric_pipeline,
            numeric_features
        )
    )

if categorical_features:

    transformers.append(
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# 23. MODEL DEFINITIONS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

    "HistGradientBoosting":
        HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE
        ),
}


# ============================================================
# 24. BASELINE MODEL COMPARISON
# ============================================================

print_section("BASELINE MODEL TRAINING")

baseline_results = []

trained_baselines = {}

for model_name, model in models.items():

    print(
        f"\nTraining: {model_name}"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )

    start = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    probabilities = pipeline.predict_proba(
        X_valid
    )[:, 1]

    metrics = calculate_metrics(
        y_valid,
        probabilities,
        threshold=0.50
    )

    metrics["Model"] = model_name
    metrics["Training_Time_Seconds"] = (
        time.time() - start
    )

    baseline_results.append(metrics)
    trained_baselines[model_name] = pipeline

    print(
        f"ROC-AUC: {metrics['ROC_AUC']:.4f} | "
        f"PR-AUC: {metrics['PR_AUC']:.4f} | "
        f"F1: {metrics['F1']:.4f}"
    )


baseline_df = pd.DataFrame(
    baseline_results
)

baseline_df = baseline_df[
    [
        "Model",
        "ROC_AUC",
        "PR_AUC",
        "Precision",
        "Recall",
        "F1",
        "Balanced_Accuracy",
        "Brier_Score",
        "Training_Time_Seconds",
    ]
]

baseline_df.to_csv(
    TABLE_DIR / "baseline_model_comparison.csv",
    index=False
)


# ============================================================
# 25. OPTUNA OPTIMIZATION
# ============================================================

print_section("OPTUNA HYPERPARAMETER OPTIMIZATION")


# Use a smaller training sample during tuning for speed.
if len(X_train) > TUNING_MAX_ROWS:

    tuning_indices = (
        np.linspace(
            0,
            len(X_train) - 1,
            TUNING_MAX_ROWS
        )
        .astype(int)
    )

    X_tune = X_train.iloc[
        tuning_indices
    ].copy()

    y_tune = y_train.iloc[
        tuning_indices
    ].copy()

else:

    X_tune = X_train.copy()
    y_tune = y_train.copy()


# ============================================================
# Optuna objective helper
# ============================================================

def build_model_with_params(model_name, trial):

    if model_name == "Logistic Regression":

        model = LogisticRegression(
            C=trial.suggest_float(
                "C",
                0.01,
                10,
                log=True
            ),
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        )

    elif model_name == "Decision Tree":

        model = DecisionTreeClassifier(
            max_depth=trial.suggest_int(
                "max_depth",
                3,
                15
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf",
                5,
                50
            ),
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    elif model_name == "Random Forest":

        model = RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators",
                100,
                250
            ),
            max_depth=trial.suggest_int(
                "max_depth",
                5,
                18
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf",
                2,
                20
            ),
            max_features=trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"]
            ),
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif model_name == "Extra Trees":

        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators",
                100,
                250
            ),
            max_depth=trial.suggest_int(
                "max_depth",
                5,
                18
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf",
                2,
                20
            ),
            max_features=trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"]
            ),
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif model_name == "HistGradientBoosting":

        model = HistGradientBoostingClassifier(
            max_iter=trial.suggest_int(
                "max_iter",
                80,
                200
            ),
            learning_rate=trial.suggest_float(
                "learning_rate",
                0.02,
                0.15,
                log=True
            ),
            max_leaf_nodes=trial.suggest_int(
                "max_leaf_nodes",
                15,
                63
            ),
            l2_regularization=trial.suggest_float(
                "l2_regularization",
                1e-3,
                10,
                log=True
            ),
            random_state=RANDOM_STATE
        )

    return model


optuna_results = []
best_params = {}
best_optuna_models = {}


# ============================================================
# NOTE:
# To keep execution practical, optimize each model using
# the fixed validation period rather than expensive nested CV.
# The final test remains untouched.
# ============================================================

for model_name in models.keys():

    print(
        f"\n{'-' * 60}"
    )

    print(
        f"OPTUNA: {model_name}"
    )

    print(
        f"{'-' * 60}"
    )

    def objective(trial):

        model = build_model_with_params(
            model_name,
            trial
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                ),
            ]
        )

        pipeline.fit(
            X_tune,
            y_tune
        )

        probabilities = pipeline.predict_proba(
            X_valid
        )[:, 1]

        return roc_auc_score(
            y_valid,
            probabilities
        )


    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        )
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=False
    )

    best_params[model_name] = study.best_params

    print(
        f"Best ROC-AUC: "
        f"{study.best_value:.5f}"
    )

    print(
        f"Best params: "
        f"{study.best_params}"
    )

    optuna_results.append(
        {
            "Model": model_name,
            "Best_ROC_AUC": study.best_value,
            "Best_Params": str(
                study.best_params
            ),
        }
    )

    # Train optimized model on complete training set
    best_model = build_model_with_params(
        model_name,
        study.best_trial
    )

    final_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                best_model
            ),
        ]
    )

    final_pipeline.fit(
        X_train,
        y_train
    )

    best_optuna_models[
        model_name
    ] = final_pipeline


optuna_df = pd.DataFrame(
    optuna_results
)

optuna_df.to_csv(
    TABLE_DIR / "optuna_best_params.csv",
    index=False
)


# ============================================================
# 26. OPTIMIZED MODEL COMPARISON
# ============================================================

print_section("OPTIMIZED MODEL EVALUATION")

optimized_results = []

validation_probabilities = {}

for model_name, pipeline in best_optuna_models.items():

    probabilities = pipeline.predict_proba(
        X_valid
    )[:, 1]

    validation_probabilities[
        model_name
    ] = probabilities

    metrics = calculate_metrics(
        y_valid,
        probabilities,
        threshold=0.50
    )

    metrics["Model"] = model_name

    optimized_results.append(
        metrics
    )

optimized_df = pd.DataFrame(
    optimized_results
)

optimized_df = optimized_df[
    [
        "Model",
        "ROC_AUC",
        "PR_AUC",
        "Precision",
        "Recall",
        "F1",
        "Balanced_Accuracy",
        "Brier_Score",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
]

optimized_df = optimized_df.sort_values(
    "ROC_AUC",
    ascending=False
)

optimized_df.to_csv(
    TABLE_DIR / "optimized_model_comparison.csv",
    index=False
)

print(
    optimized_df.to_string(
        index=False
    )
)


# ============================================================
# 27. SELECT BEST MODEL
# ============================================================

best_model_name = optimized_df.iloc[
    0
]["Model"]

best_pipeline = best_optuna_models[
    best_model_name
]

print(
    f"\nSelected model: "
    f"{best_model_name}"
)


# ============================================================
# 28. THRESHOLD OPTIMIZATION
# ============================================================

print_section("THRESHOLD OPTIMIZATION")

best_valid_probabilities = (
    validation_probabilities[
        best_model_name
    ]
)

threshold_rows = []

for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    predictions = (
        best_valid_probabilities
        >= threshold
    ).astype(int)

    threshold_rows.append(
        {
            "threshold": threshold,
            "precision": precision_score(
                y_valid,
                predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_valid,
                predictions,
                zero_division=0
            ),
            "f1": f1_score(
                y_valid,
                predictions,
                zero_division=0
            ),
            "balanced_accuracy":
                balanced_accuracy_score(
                    y_valid,
                    predictions
                ),
        }
    )

threshold_df = pd.DataFrame(
    threshold_rows
)

# Select threshold using maximum F1.
best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

BEST_THRESHOLD = float(
    best_threshold_row["threshold"]
)

threshold_df.to_csv(
    TABLE_DIR / "threshold_analysis.csv",
    index=False
)

print(
    f"Selected threshold: "
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Validation F1: "
    f"{best_threshold_row['f1']:.4f}"
)


with open(
    REPORT_DIR / "threshold.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "selected_model": best_model_name,
            "threshold": BEST_THRESHOLD,
            "selection_metric": "F1",
        },
        f,
        indent=4
    )


# ============================================================
# 29. FINAL TEST EVALUATION
# ============================================================

print_section("FINAL TEST EVALUATION")

test_probabilities = best_pipeline.predict_proba(
    X_test
)[:, 1]

test_metrics = calculate_metrics(
    y_test,
    test_probabilities,
    threshold=BEST_THRESHOLD
)

test_results = pd.DataFrame(
    [
        {
            "Model": best_model_name,
            **test_metrics
        }
    ]
)

test_results.to_csv(
    TABLE_DIR / "final_test_metrics.csv",
    index=False
)

print(
    test_results.to_string(
        index=False
    )
)


# ============================================================
# 30. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    (
        test_probabilities
        >= BEST_THRESHOLD
    ).astype(int),
    labels=[0, 1]
)

plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "Final Model Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["Non-Retained", "Retained"]
)

plt.yticks(
    [0, 1],
    ["Non-Retained", "Retained"]
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

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# 31. ROC CURVE
# ============================================================

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(
    y_test,
    test_probabilities
)

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {test_metrics['ROC_AUC']:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Final Model ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_roc_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# 32. PRECISION-RECALL CURVE
# ============================================================

from sklearn.metrics import precision_recall_curve

precision_values, recall_values, _ = (
    precision_recall_curve(
        y_test,
        test_probabilities
    )
)

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Final Model Precision-Recall Curve"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_precision_recall_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# 33. CALIBRATION CURVE
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_test,
    test_probabilities,
    n_bins=10,
    strategy="quantile"
)

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Observed Frequency"
)

plt.title(
    "Final Model Calibration"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_calibration_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# 34. PERMUTATION IMPORTANCE
# ============================================================
#
# We calculate this on raw validation features.
# This avoids relying on model-specific feature names.
# ============================================================

print_section("FEATURE IMPORTANCE")

try:

    perm = permutation_importance(
        best_pipeline,
        X_valid,
        y_valid,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_valid.columns,
            "importance_mean":
                perm.importances_mean,
            "importance_std":
                perm.importances_std,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance_mean",
            ascending=False
        )
    )

    importance_df.to_csv(
        TABLE_DIR / "feature_importance.csv",
        index=False
    )

    top_importance = (
        importance_df
        .head(15)
        .sort_values(
            "importance_mean"
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.barh(
        top_importance["feature"],
        top_importance["importance_mean"]
    )

    plt.xlabel(
        "Permutation Importance"
    )

    plt.title(
        "Top Predictive Features"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "05_feature_importance.png",
        dpi=150
    )

    plt.close()

except Exception as e:

    print(
        "Feature importance could not be calculated:"
    )

    print(e)


# ============================================================
# 35. CUSTOMER-LEVEL PREDICTIONS
# ============================================================

print_section("GENERATING CUSTOMER PREDICTIONS")

predictions = test[
    [
        CUSTOMER_COL,
        "prediction_point"
    ]
].copy()

predictions[
    "retention_probability"
] = test_probabilities

predictions[
    "predicted_retention"
] = (
    test_probabilities
    >= BEST_THRESHOLD
).astype(int)


def assign_retention_segment(probability):

    if probability >= 0.70:
        return "High Retention Probability"

    elif probability >= 0.40:
        return "Medium Retention Probability"

    return "Low Retention Probability"


predictions[
    "retention_segment"
] = predictions[
    "retention_probability"
].apply(
    assign_retention_segment
)

predictions.to_csv(
    TABLE_DIR / "customer_predictions.csv",
    index=False
)


# ============================================================
# 36. SAVE FINAL MODEL
# ============================================================

print_section("SAVING MODEL")

joblib.dump(
    best_pipeline,
    MODEL_DIR / "best_retention_model.joblib"
)

print(
    f"Saved:\n"
    f"{MODEL_DIR / 'best_retention_model.joblib'}"
)


# ============================================================
# 37. SAVE MODEL INFORMATION
# ============================================================

model_information = {

    "project":
        "Customer Cohort Retention Analysis",

    "part":
        "Part 2 - Customer Retention Prediction",

    "selected_model":
        best_model_name,

    "target":
        "retained",

    "retention_definition":
        "At least one purchase during the future "
        "prediction window",

    "observation_window_days":
        OBSERVATION_WINDOW_DAYS,

    "future_window_days":
        FUTURE_WINDOW_DAYS,

    "test_size":
        TEST_SIZE,

    "optuna_trials_per_model":
        N_TRIALS,

    "random_state":
        RANDOM_STATE,

    "selected_threshold":
        BEST_THRESHOLD,

    "final_test_roc_auc":
        float(test_metrics["ROC_AUC"]),

    "final_test_pr_auc":
        float(test_metrics["PR_AUC"]),

    "final_test_precision":
        float(test_metrics["Precision"]),

    "final_test_recall":
        float(test_metrics["Recall"]),

    "final_test_f1":
        float(test_metrics["F1"]),

    "final_test_balanced_accuracy":
        float(
            test_metrics[
                "Balanced_Accuracy"
            ]
        ),
}


with open(
    REPORT_DIR / "model_information.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_information,
        f,
        indent=4
    )


# ============================================================
# 38. FINAL REPORT
# ============================================================

runtime = time.time() - start_time

report = f"""
CUSTOMER COHORT RETENTION ANALYSIS
PART 2 — CUSTOMER RETENTION PREDICTION
======================================

DATASET
-------
Path:
{DATA_PATH}

Final prediction rows:
{len(customer_features):,}

Final raw features:
{X_train.shape[1]}

RETENTION DEFINITION
--------------------
A customer is considered retained if they make at least
one purchase during the future prediction window.

Observation window:
{OBSERVATION_WINDOW_DAYS} days

Future prediction window:
{FUTURE_WINDOW_DAYS} days

DATA SPLIT
----------
Train:
{len(train):,}

Validation:
{len(validation):,}

Test:
{len(test):,}

Target distribution
-------------------
Retained:
{int(y_train.sum()):,} in training

Non-retained:
{int((y_train == 0).sum()):,} in training

MODEL
-----
Selected model:
{best_model_name}

Optuna trials per model:
{N_TRIALS}

Threshold:
{BEST_THRESHOLD:.2f}

FINAL TEST PERFORMANCE
----------------------
ROC-AUC:
{test_metrics['ROC_AUC']:.4f}

PR-AUC:
{test_metrics['PR_AUC']:.4f}

Precision:
{test_metrics['Precision']:.4f}

Recall:
{test_metrics['Recall']:.4f}

F1:
{test_metrics['F1']:.4f}

Balanced Accuracy:
{test_metrics['Balanced_Accuracy']:.4f}

Brier Score:
{test_metrics['Brier_Score']:.4f}

CONFUSION MATRIX
----------------
TN: {test_metrics['TN']}
FP: {test_metrics['FP']}
FN: {test_metrics['FN']}
TP: {test_metrics['TP']}

RUNTIME
-------
{runtime / 60:.2f} minutes

LEAKAGE PREVENTION
------------------
Features were constructed only from the observation window.

Future order counts were used only to construct the target
and were excluded from model features.

The final test set was not used for:
- Optuna optimization
- threshold selection
- feature selection
- model selection
"""


with open(
    REPORT_DIR / "final_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# 39. FINAL SUMMARY
# ============================================================

print_section("PART 2 COMPLETED")

print(
    f"""
Project:
Customer Cohort Retention Analysis

Part:
Part 2 — Retention Prediction

Rows:
{len(customer_features):,}

Features:
{X_train.shape[1]}

Selected model:
{best_model_name}

Threshold:
{BEST_THRESHOLD:.2f}

Final Test ROC-AUC:
{test_metrics['ROC_AUC']:.4f}

Final Test PR-AUC:
{test_metrics['PR_AUC']:.4f}

Final Test F1:
{test_metrics['F1']:.4f}

Outputs:
{OUTPUT_DIR}

Runtime:
{runtime / 60:.2f} minutes
"""
)

print("=" * 70)
print("DONE")
print("=" * 70)