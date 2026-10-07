# ============================================================
# OLIST E-COMMERCE DATASET
# COMPLETE PROFESSIONAL EDA + STATISTICAL TESTING
# FAST + CLEAN + GITHUB READY
# ============================================================

# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import (
    shapiro,
    normaltest,
    levene,
    bartlett,
    ttest_ind,
    mannwhitneyu,
    f_oneway,
    kruskal,
    chi2_contingency,
    pearsonr,
    spearmanr
)

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

MAX_SAMPLE = 5000
MAX_PLOT_SAMPLE = 20000
MAX_NORMALITY_SAMPLE = 5000

Z_THRESHOLD = 3

SKEW_HIGH = 1
KURTOSIS_HIGH = 3

MAX_DISTRIBUTION_VARS = 8
MAX_BOXPLOT_VARS = 10
MAX_CATEGORICAL_VARS = 6
MAX_CORRELATION_VARS = 12
MAX_QQ_VARS = 6
MAX_GROUP_VARS = 4
MAX_TEST_PAIRS = 25
MAX_VIF_VARS = 15

# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path("C:\\Users\\Hp\\Documents\\New folder\\olist_6_files_merged.csv").resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PLOT STYLE
# ============================================================

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    """Save plot with clean layout."""
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / filename,
        bbox_inches="tight"
    )
    plt.close()


def safe_sample(series, n=MAX_SAMPLE):
    """Return random sample without exceeding available rows."""
    series = series.dropna()

    if len(series) > n:
        return series.sample(
            n=n,
            random_state=RANDOM_STATE
        )

    return series


def interpretation_strength(value):
    """Interpret absolute effect/correlation strength."""
    value = abs(value)

    if value < 0.10:
        return "Negligible"
    elif value < 0.30:
        return "Small"
    elif value < 0.50:
        return "Moderate"
    else:
        return "Strong"


def cramers_v(table):
    """Bias-corrected Cramer's V."""
    chi2, p, dof, expected = chi2_contingency(table)

    n = table.sum().sum()

    if n <= 1:
        return np.nan

    phi2 = chi2 / n
    rows, cols = table.shape

    phi2corr = max(
        0,
        phi2 - ((cols - 1) * (rows - 1)) / (n - 1)
    )

    rcorr = rows - (
        (rows - 1) ** 2 / (n - 1)
    )

    ccorr = cols - (
        (cols - 1) ** 2 / (n - 1)
    )

    denominator = min(ccorr - 1, rcorr - 1)

    if denominator <= 0:
        return np.nan

    return np.sqrt(phi2corr / denominator)


def cohens_d(x, y):
    """Calculate Cohen's d."""
    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled_sd = np.sqrt(
        ((nx - 1) * vx + (ny - 1) * vy)
        / (nx + ny - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd


def rank_biserial(x, y):
    """Signed rank-biserial effect size."""
    x = np.asarray(x)
    y = np.asarray(y)

    n1 = len(x)
    n2 = len(y)

    if n1 == 0 or n2 == 0:
        return np.nan

    result = mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    )

    u = result.statistic

    return (2 * u) / (n1 * n2) - 1


def eta_squared(groups):
    """Calculate eta squared."""
    all_values = np.concatenate(groups)

    grand_mean = np.mean(all_values)

    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in groups
    )

    ss_total = np.sum(
        (all_values - grand_mean) ** 2
    )

    if ss_total == 0:
        return np.nan

    return ss_between / ss_total


def apply_fdr(df, p_column="p_value"):
    """Benjamini-Hochberg FDR correction."""
    if df.empty or p_column not in df.columns:
        return df

    result = df.copy()

    p_values = result[p_column].fillna(1).values

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    n = len(sorted_p)

    adjusted = np.empty(n)

    previous = 1.0

    for i in range(n - 1, -1, -1):

        rank = i + 1

        value = (
            sorted_p[i] * n / rank
        )

        previous = min(previous, value)

        adjusted[i] = previous

    fdr = np.empty(n)

    fdr[order] = adjusted

    result["fdr_p_value"] = fdr

    result["significant_fdr"] = (
        result["fdr_p_value"] < 0.05
    )

    return result


def classify_skew(value):
    value = abs(value)

    if value <= 0.5:
        return "Approximately symmetric"
    elif value <= 1:
        return "Moderately skewed"
    else:
        return "Highly skewed"


def classify_kurtosis(value):
    value = abs(value)

    if value <= 1:
        return "Typical"
    elif value <= 3:
        return "Moderately heavy/light-tailed"
    else:
        return "Strongly heavy/light-tailed"


def normality_interpretation(skew_value, kurtosis_value):
    if (
        abs(skew_value) <= 1
        and abs(kurtosis_value) <= 3
    ):
        return "Approximately normal"

    return "Not approximately normal"


def find_dataset():
    """Automatically locate CSV dataset."""

    possible_names = [
        "olist_6_files_merged.csv",
        "olist_6_files_merged(1).csv",
        "olist.csv"
    ]

    search_locations = [
        PROJECT_ROOT,
        DATA_DIR
    ]

    for location in search_locations:

        for name in possible_names:

            path = location / name

            if path.exists():
                return path

    csv_files = list(
        PROJECT_ROOT.glob("*.csv")
    ) + list(
        DATA_DIR.glob("*.csv")
    )

    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        "No CSV dataset found. "
        "Put your OLIST CSV inside the project folder "
        "or inside the data folder."
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("OLIST DATASET - COMPLETE EDA + STATISTICAL TESTING")
print("=" * 80)

dataset_path = find_dataset()

print(f"\nDataset: {dataset_path}")

df = pd.read_csv(
    dataset_path,
    low_memory=False
)

print(f"Loaded shape: {df.shape}")


# ============================================================
# 5. BASIC CLEANING
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

# Replace infinite values
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 6. DATETIME DETECTION
# ============================================================

datetime_cols = []

for col in df.select_dtypes(
    include="object"
).columns:

    sample = df[col].dropna().astype(str)

    if len(sample) == 0:
        continue

    sample = sample.head(1000)

    parsed = pd.to_datetime(
        sample,
        errors="coerce"
    )

    success_rate = parsed.notna().mean()

    if success_rate >= 0.80:

        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        datetime_cols.append(col)


# ============================================================
# 7. IDENTIFY DATA TYPES
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

datetime_cols = df.select_dtypes(
    include=["datetime"]
).columns.tolist()


# ============================================================
# 8. DATASET OVERVIEW
# ============================================================

print("\n" + "=" * 80)
print("1. DATASET OVERVIEW")
print("=" * 80)

overview = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Numeric Columns",
        "Categorical Columns",
        "Datetime Columns",
        "Duplicate Rows",
        "Total Missing Cells"
    ],
    "Value": [
        len(df),
        df.shape[1],
        len(numeric_cols),
        len(categorical_cols),
        len(datetime_cols),
        df.duplicated().sum(),
        int(df.isna().sum().sum())
    ]
})

print(overview.to_string(index=False))

overview.to_csv(
    TABLE_DIR / "01_dataset_overview.csv",
    index=False
)


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("2. MISSING VALUES")
print("=" * 80)

missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values
})

missing["missing_percent"] = (
    missing["missing_count"]
    / len(df)
    * 100
)

missing = missing.sort_values(
    "missing_percent",
    ascending=False
)

print(
    missing.head(20).to_string(index=False)
)

missing.to_csv(
    TABLE_DIR / "02_missing_values.csv",
    index=False
)


# ============================================================
# 10. DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

duplicate_result = pd.DataFrame({
    "duplicate_rows": [duplicate_count],
    "duplicate_percent": [
        duplicate_count / len(df) * 100
    ]
})

duplicate_result.to_csv(
    TABLE_DIR / "03_duplicates.csv",
    index=False
)


# ============================================================
# 11. CONSTANT + NEAR-ZERO VARIANCE
# ============================================================

constant_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=True) <= 1
]

near_zero_cols = []

for col in df.columns:

    counts = df[col].value_counts(
        dropna=False,
        normalize=True
    )

    if len(counts) > 0 and counts.iloc[0] >= 0.99:
        near_zero_cols.append(col)

variance_quality = pd.DataFrame({
    "column": df.columns,
    "is_constant": [
        col in constant_cols
        for col in df.columns
    ],
    "near_zero_variance": [
        col in near_zero_cols
        for col in df.columns
    ]
})

variance_quality.to_csv(
    TABLE_DIR / "04_variance_quality.csv",
    index=False
)

print("\nConstant columns:")
print(constant_cols)


# ============================================================
# 12. ID-LIKE COLUMN DETECTION
# ============================================================

id_like_cols = []

for col in df.columns:

    name = col.lower()

    unique_ratio = (
        df[col].nunique(dropna=True)
        / max(df[col].notna().sum(), 1)
    )

    if (
        unique_ratio >= 0.95
        or name.endswith("_id")
        or name.endswith("_key")
        or name == "id"
        or "uuid" in name
    ):
        id_like_cols.append(col)

id_like_cols = list(dict.fromkeys(id_like_cols))

pd.DataFrame({
    "id_like_column": id_like_cols
}).to_csv(
    TABLE_DIR / "05_id_like_columns.csv",
    index=False
)

print("\nID-like columns:")
print(id_like_cols)


# ============================================================
# 13. ANALYSIS NUMERIC COLUMNS
# ============================================================

analysis_numeric_cols = [
    col for col in numeric_cols
    if col not in constant_cols
    and col not in id_like_cols
]


# ============================================================
# 14. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 80)

if analysis_numeric_cols:

    descriptive = df[
        analysis_numeric_cols
    ].describe().T

    descriptive["variance"] = (
        df[analysis_numeric_cols]
        .var()
    )

    descriptive["IQR"] = (
        df[analysis_numeric_cols]
        .quantile(0.75)
        -
        df[analysis_numeric_cols]
        .quantile(0.25)
    )

    descriptive["missing_count"] = (
        df[analysis_numeric_cols]
        .isna()
        .sum()
    )

    descriptive["missing_percent"] = (
        descriptive["missing_count"]
        / len(df)
        * 100
    )

    descriptive.to_csv(
        TABLE_DIR / "06_descriptive_statistics.csv"
    )

    print(
        descriptive.head(20).to_string()
    )


# ============================================================
# 15. VARIANCE ANALYSIS
# ============================================================

variance_table = pd.DataFrame({
    "column": analysis_numeric_cols,
    "mean": [
        df[col].mean()
        for col in analysis_numeric_cols
    ],
    "variance": [
        df[col].var()
        for col in analysis_numeric_cols
    ],
    "standard_deviation": [
        df[col].std()
        for col in analysis_numeric_cols
    ],
    "IQR": [
        df[col].quantile(0.75)
        - df[col].quantile(0.25)
        for col in analysis_numeric_cols
    ]
})

variance_table = variance_table.sort_values(
    "variance",
    ascending=False
)

variance_table.to_csv(
    TABLE_DIR / "07_variance_analysis.csv",
    index=False
)


# ============================================================
# 16. OUTLIER ANALYSIS
# ============================================================

outlier_rows = []

for col in analysis_numeric_cols:

    series = df[col].dropna()

    if len(series) < 5:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        iqr_outliers = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        iqr_outliers = (
            (series < lower)
            | (series > upper)
        ).sum()

    mean = series.mean()
    std = series.std()

    if std == 0:
        z_outliers = 0
    else:
        z_outliers = (
            np.abs(
                (series - mean) / std
            ) > Z_THRESHOLD
        ).sum()

    outlier_rows.append({
        "column": col,
        "IQR_outliers": int(iqr_outliers),
        "IQR_outlier_percent":
            iqr_outliers / len(series) * 100,
        "Z_score_outliers": int(z_outliers),
        "Z_outlier_percent":
            z_outliers / len(series) * 100
    })

outliers = pd.DataFrame(outlier_rows)

outliers = outliers.sort_values(
    "IQR_outlier_percent",
    ascending=False
)

outliers.to_csv(
    TABLE_DIR / "08_outlier_analysis.csv",
    index=False
)


# ============================================================
# 17. SKEWNESS
# ============================================================

skew_rows = []

for col in analysis_numeric_cols:

    series = df[col].dropna()

    if len(series) < 3:
        continue

    value = series.skew()

    skew_rows.append({
        "column": col,
        "skewness": value,
        "classification": classify_skew(value),
        "high_skew": abs(value) > SKEW_HIGH
    })

skewness = pd.DataFrame(skew_rows)

skewness = skewness.sort_values(
    "skewness",
    key=lambda x: x.abs(),
    ascending=False
)

skewness.to_csv(
    TABLE_DIR / "09_skewness_analysis.csv",
    index=False
)

high_skew_cols = skewness.loc[
    skewness["high_skew"],
    "column"
].tolist()

print("\nHighly skewed variables:")
print(high_skew_cols)


# ============================================================
# 18. YEO-JOHNSON TRANSFORMATION
# ============================================================

transformation_rows = []

transformed_data = {}

for col in high_skew_cols:

    series = df[col].copy()

    valid = series.dropna()

    if len(valid) < 20:
        continue

    try:

        transformer = PowerTransformer(
            method="yeo-johnson",
            standardize=False
        )

        transformed_values = transformer.fit_transform(
            valid.values.reshape(-1, 1)
        ).flatten()

        before_skew = valid.skew()
        after_skew = pd.Series(
            transformed_values
        ).skew()

        transformed_data[col] = pd.Series(
            transformed_values,
            index=valid.index
        )

        transformation_rows.append({
            "column": col,
            "before_skewness": before_skew,
            "after_skewness": after_skew,
            "absolute_improvement":
                abs(before_skew)
                - abs(after_skew),
            "improved":
                abs(after_skew)
                < abs(before_skew)
        })

    except Exception as e:

        transformation_rows.append({
            "column": col,
            "before_skewness":
                valid.skew(),
            "after_skewness": np.nan,
            "absolute_improvement": np.nan,
            "improved": False
        })

transformation_results = pd.DataFrame(
    transformation_rows
)

transformation_results.to_csv(
    TABLE_DIR / "10_yeo_johnson_validation.csv",
    index=False
)


# ============================================================
# 19. KURTOSIS
# ============================================================

kurtosis_rows = []

for col in analysis_numeric_cols:

    series = df[col].dropna()

    if len(series) < 4:
        continue

    value = series.kurt()

    kurtosis_rows.append({
        "column": col,
        "excess_kurtosis": value,
        "classification":
            classify_kurtosis(value),
        "high_kurtosis":
            abs(value) > KURTOSIS_HIGH
    })

kurtosis_table = pd.DataFrame(
    kurtosis_rows
)

kurtosis_table = kurtosis_table.sort_values(
    "excess_kurtosis",
    key=lambda x: x.abs(),
    ascending=False
)

kurtosis_table.to_csv(
    TABLE_DIR / "11_kurtosis_analysis.csv",
    index=False
)


# ============================================================
# 20. NORMALITY TESTING
# ============================================================

print("\n" + "=" * 80)
print("4. NORMALITY TESTING")
print("=" * 80)

normality_candidates = [
    col for col in analysis_numeric_cols
    if df[col].nunique(dropna=True) >= 20
]

# Select representative variables using skewness
if normality_candidates:

    normality_candidates = sorted(
        normality_candidates,
        key=lambda c: abs(
            df[c].skew()
        ),
        reverse=True
    )[:MAX_QQ_VARS]

normality_rows = []

for col in normality_candidates:

    series = safe_sample(
        df[col],
        MAX_NORMALITY_SAMPLE
    )

    if len(series) < 20:
        continue

    try:

        shapiro_stat, shapiro_p = shapiro(
            series
        )

        normal_stat, normal_p = normaltest(
            series
        )

        skew_value = series.skew()
        kurt_value = series.kurt()

        normality_rows.append({
            "column": col,
            "sample_size": len(series),
            "shapiro_p_value": shapiro_p,
            "dagostino_p_value": normal_p,
            "skewness": skew_value,
            "excess_kurtosis": kurt_value,
            "interpretation":
                normality_interpretation(
                    skew_value,
                    kurt_value
                )
        })

    except Exception:
        pass

normality_results = pd.DataFrame(
    normality_rows
)

normality_results.to_csv(
    TABLE_DIR / "12_normality_tests.csv",
    index=False
)


# ============================================================
# 21. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("5. CORRELATION ANALYSIS")
print("=" * 80)

correlation_candidates = [
    col for col in analysis_numeric_cols
    if df[col].nunique(dropna=True) > 5
]

# Select variables with largest variance
correlation_candidates = sorted(
    correlation_candidates,
    key=lambda c: df[c].var(),
    reverse=True
)[:MAX_CORRELATION_VARS]

if len(correlation_candidates) >= 2:

    pearson_matrix = df[
        correlation_candidates
    ].corr(
        method="pearson"
    )

    spearman_matrix = df[
        correlation_candidates
    ].corr(
        method="spearman"
    )

    pearson_matrix.to_csv(
        TABLE_DIR / "13_pearson_correlation.csv"
    )

    spearman_matrix.to_csv(
        TABLE_DIR / "14_spearman_correlation.csv"
    )

    pair_rows = []

    for i in range(
        len(correlation_candidates)
    ):

        for j in range(i + 1, len(
            correlation_candidates
        )):

            x_col = correlation_candidates[i]
            y_col = correlation_candidates[j]

            temp = df[
                [x_col, y_col]
            ].dropna()

            if len(temp) < 10:
                continue

            try:

                p_r, p_p = pearsonr(
                    temp[x_col],
                    temp[y_col]
                )

                s_r, s_p = spearmanr(
                    temp[x_col],
                    temp[y_col]
                )

                pair_rows.append({
                    "variable_1": x_col,
                    "variable_2": y_col,
                    "pearson_r": p_r,
                    "pearson_p": p_p,
                    "spearman_r": s_r,
                    "spearman_p": s_p,
                    "pearson_strength":
                        interpretation_strength(p_r),
                    "spearman_strength":
                        interpretation_strength(s_r)
                })

            except Exception:
                pass

    correlation_pairs = pd.DataFrame(
        pair_rows
    )

    if not correlation_pairs.empty:

        correlation_pairs["abs_pearson"] = (
            correlation_pairs["pearson_r"].abs()
        )

        correlation_pairs = correlation_pairs.sort_values(
            "abs_pearson",
            ascending=False
        )

        correlation_pairs.to_csv(
            TABLE_DIR / "15_correlation_pairs.csv",
            index=False
        )


# ============================================================
# 22. GROUPING / DEPENDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("6. GROUP MEAN / DEPENDENCE ANALYSIS")
print("=" * 80)

group_candidates = []

for col in categorical_cols:

    if col in id_like_cols:
        continue

    unique_count = df[col].nunique(
        dropna=True
    )

    if 2 <= unique_count <= 15:
        group_candidates.append(col)


# Also consider low-cardinality numeric columns
for col in analysis_numeric_cols:

    unique_count = df[col].nunique(
        dropna=True
    )

    if 2 <= unique_count <= 10:
        group_candidates.append(col)


group_candidates = list(
    dict.fromkeys(group_candidates)
)

group_candidates = group_candidates[
    :MAX_GROUP_VARS
]

# Choose a meaningful numeric target
target_candidates = [
    col for col in analysis_numeric_cols
    if df[col].nunique(dropna=True) > 10
]

target_candidates = sorted(
    target_candidates,
    key=lambda c: df[c].var(),
    reverse=True
)

group_target = (
    target_candidates[0]
    if target_candidates
    else None
)

group_mean_rows = []

if group_target is not None:

    for group_col in group_candidates:

        temp = df[
            [group_col, group_target]
        ].dropna()

        if temp.empty:
            continue

        grouped = temp.groupby(
            group_col,
            observed=True
        )[group_target].agg(
            ["count", "mean", "median", "std"]
        ).reset_index()

        for _, row in grouped.iterrows():

            group_mean_rows.append({
                "group_variable":
                    group_col,
                "group":
                    row[group_col],
                "target":
                    group_target,
                "count":
                    row["count"],
                "mean":
                    row["mean"],
                "median":
                    row["median"],
                "std":
                    row["std"]
            })

group_means = pd.DataFrame(
    group_mean_rows
)

group_means.to_csv(
    TABLE_DIR / "16_group_means.csv",
    index=False
)


# ============================================================
# 23. STATISTICAL TESTING
# ============================================================

print("\n" + "=" * 80)
print("7. STATISTICAL TESTING")
print("=" * 80)

test_rows = []


# ------------------------------------------------------------
# 23A. CORRELATION TESTS
# ------------------------------------------------------------

if "correlation_pairs" in locals():

    top_pairs = correlation_pairs.head(
        MAX_TEST_PAIRS
    )

    for _, row in top_pairs.iterrows():

        test_rows.append({
            "test": "Pearson correlation",
            "variable_1":
                row["variable_1"],
            "variable_2":
                row["variable_2"],
            "statistic":
                row["pearson_r"],
            "p_value":
                row["pearson_p"],
            "effect_size":
                row["pearson_r"],
            "effect_type":
                "Correlation"
        })

        test_rows.append({
            "test": "Spearman correlation",
            "variable_1":
                row["variable_1"],
            "variable_2":
                row["variable_2"],
            "statistic":
                row["spearman_r"],
            "p_value":
                row["spearman_p"],
            "effect_size":
                row["spearman_r"],
            "effect_type":
                "Correlation"
        })


# ------------------------------------------------------------
# 23B. GROUP TESTS
# ------------------------------------------------------------

if group_target is not None:

    for group_col in group_candidates:

        temp = df[
            [group_col, group_target]
        ].dropna()

        groups = []

        labels = []

        for label, subset in temp.groupby(
            group_col,
            observed=True
        ):

            values = subset[
                group_target
            ].values

            if len(values) >= 10:

                groups.append(values)
                labels.append(label)

        if len(groups) < 2:
            continue


        # ====================================================
        # TWO GROUPS
        # ====================================================

        if len(groups) == 2:

            g1, g2 = groups

            normal_flags = []

            for g in groups:

                sample = (
                    g
                    if len(g) <= 5000
                    else np.random.default_rng(
                        RANDOM_STATE
                    ).choice(
                        g,
                        5000,
                        replace=False
                    )
                )

                try:
                    p = shapiro(sample).pvalue
                    normal_flags.append(
                        p >= 0.05
                    )
                except Exception:
                    normal_flags.append(False)

            try:

                levene_stat, levene_p = levene(
                    g1,
                    g2,
                    center="median"
                )

                test_rows.append({
                    "test":
                        "Levene variance equality",
                    "variable_1":
                        group_col,
                    "variable_2":
                        group_target,
                    "statistic":
                        levene_stat,
                    "p_value":
                        levene_p,
                    "effect_size":
                        np.nan,
                    "effect_type":
                        "Variance"
                })

            except Exception:
                levene_p = 0

            # Welch t-test when approximately normal
            if all(normal_flags):

                statistic, p_value = ttest_ind(
                    g1,
                    g2,
                    equal_var=False
                )

                effect = cohens_d(
                    g1,
                    g2
                )

                test_name = (
                    "Welch independent t-test"
                )

                effect_type = "Cohen's d"

            else:

                statistic, p_value = mannwhitneyu(
                    g1,
                    g2,
                    alternative="two-sided"
                )

                effect = rank_biserial(
                    g1,
                    g2
                )

                test_name = (
                    "Mann-Whitney U test"
                )

                effect_type = (
                    "Rank-biserial correlation"
                )

            test_rows.append({
                "test": test_name,
                "variable_1":
                    group_col,
                "variable_2":
                    group_target,
                "statistic":
                    statistic,
                "p_value":
                    p_value,
                "effect_size":
                    effect,
                "effect_type":
                    effect_type
            })


        # ====================================================
        # MORE THAN TWO GROUPS
        # ====================================================

        else:

            normal_flags = []

            for g in groups:

                sample = (
                    g
                    if len(g) <= 5000
                    else np.random.default_rng(
                        RANDOM_STATE
                    ).choice(
                        g,
                        5000,
                        replace=False
                    )
                )

                try:
                    p = shapiro(sample).pvalue
                    normal_flags.append(
                        p >= 0.05
                    )
                except Exception:
                    normal_flags.append(False)

            try:

                levene_stat, levene_p = levene(
                    *groups,
                    center="median"
                )

                test_rows.append({
                    "test":
                        "Levene variance equality",
                    "variable_1":
                        group_col,
                    "variable_2":
                        group_target,
                    "statistic":
                        levene_stat,
                    "p_value":
                        levene_p,
                    "effect_size":
                        np.nan,
                    "effect_type":
                        "Variance"
                })

            except Exception:
                levene_p = 0

            if all(normal_flags):

                statistic, p_value = f_oneway(
                    *groups
                )

                effect = eta_squared(
                    groups
                )

                test_name = "One-way ANOVA"
                effect_type = "Eta squared"

            else:

                statistic, p_value = kruskal(
                    *groups
                )

                effect = eta_squared(
                    groups
                )

                test_name = "Kruskal-Wallis test"
                effect_type = "Eta squared"

            test_rows.append({
                "test": test_name,
                "variable_1":
                    group_col,
                "variable_2":
                    group_target,
                "statistic":
                    statistic,
                "p_value":
                    p_value,
                "effect_size":
                    effect,
                "effect_type":
                    effect_type
            })


# ============================================================
# 24. CHI-SQUARE + CRAMER'S V
# ============================================================

chi_candidates = [
    col for col in categorical_cols
    if col not in id_like_cols
    and 2 <= df[col].nunique(dropna=True) <= 10
]

chi_candidates = chi_candidates[:8]

for i in range(len(chi_candidates)):

    for j in range(i + 1, len(chi_candidates)):

        col1 = chi_candidates[i]
        col2 = chi_candidates[j]

        table = pd.crosstab(
            df[col1],
            df[col2]
        )

        if (
            table.shape[0] < 2
            or table.shape[1] < 2
        ):
            continue

        try:

            chi2, p_value, dof, expected = (
                chi2_contingency(table)
            )

            effect = cramers_v(table)

            test_rows.append({
                "test":
                    "Chi-square + Cramer's V",
                "variable_1":
                    col1,
                "variable_2":
                    col2,
                "statistic":
                    chi2,
                "p_value":
                    p_value,
                "effect_size":
                    effect,
                "effect_type":
                    "Cramer's V"
            })

        except Exception:
            pass


# ============================================================
# 25. FINAL STATISTICAL RESULTS
# ============================================================

statistical_tests = pd.DataFrame(
    test_rows
)

if not statistical_tests.empty:

    statistical_tests = apply_fdr(
        statistical_tests,
        "p_value"
    )

    statistical_tests["significant_raw"] = (
        statistical_tests["p_value"] < 0.05
    )

    statistical_tests["effect_magnitude"] = (
        statistical_tests["effect_size"]
        .abs()
        .apply(interpretation_strength)
    )

    statistical_tests.to_csv(
        TABLE_DIR / "17_statistical_tests.csv",
        index=False
    )

    print(
        statistical_tests.head(30)
        .to_string(index=False)
    )


# ============================================================
# 26. VIF MULTICOLLINEARITY
# ============================================================

print("\n" + "=" * 80)
print("8. MULTICOLLINEARITY - VIF")
print("=" * 80)

vif_candidates = [
    col for col in analysis_numeric_cols
    if df[col].nunique(dropna=True) > 5
]

vif_candidates = sorted(
    vif_candidates,
    key=lambda c: df[c].var(),
    reverse=True
)[:MAX_VIF_VARS]

if len(vif_candidates) >= 2:

    vif_data = df[
        vif_candidates
    ].copy()

    vif_data = vif_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    vif_data = vif_data.fillna(
        vif_data.median()
    )

    # Remove zero variance columns
    vif_data = vif_data.loc[
        :,
        vif_data.nunique() > 1
    ]

    vif_rows = []

    for col in vif_data.columns:

        y = vif_data[col].values

        X = vif_data.drop(
            columns=[col]
        )

        if X.shape[1] == 0:
            vif_value = 1.0

        else:

            model = LinearRegression()

            model.fit(X, y)

            r2 = model.score(
                X,
                y
            )

            if r2 >= 0.999999:
                vif_value = np.inf
            else:
                vif_value = 1 / (1 - r2)

        if vif_value < 5:
            interpretation = "Low"
        elif vif_value < 10:
            interpretation = "Moderate"
        else:
            interpretation = "High"

        vif_rows.append({
            "feature": col,
            "VIF": vif_value,
            "interpretation":
                interpretation
        })

    vif_results = pd.DataFrame(
        vif_rows
    ).sort_values(
        "VIF",
        ascending=False
    )

    vif_results.to_csv(
        TABLE_DIR / "18_vif_analysis.csv",
        index=False
    )

    print(
        vif_results.to_string(index=False)
    )


# ============================================================
# 27. BEFORE / AFTER TRANSFORMATION VALIDATION
# ============================================================

validation_rows = []

for col, transformed_series in transformed_data.items():

    before = df[col].dropna()

    after = transformed_series.dropna()

    before_q1 = before.quantile(0.25)
    before_q3 = before.quantile(0.75)
    before_iqr = before_q3 - before_q1

    after_q1 = after.quantile(0.25)
    after_q3 = after.quantile(0.75)
    after_iqr = after_q3 - after_q1

    if before_iqr != 0:

        before_outliers = (
            (before < before_q1 - 1.5 * before_iqr)
            |
            (before > before_q3 + 1.5 * before_iqr)
        ).sum()

    else:
        before_outliers = 0

    if after_iqr != 0:

        after_outliers = (
            (after < after_q1 - 1.5 * after_iqr)
            |
            (after > after_q3 + 1.5 * after_iqr)
        ).sum()

    else:
        after_outliers = 0

    validation_rows.append({
        "column": col,
        "before_skewness":
            before.skew(),
        "after_skewness":
            after.skew(),
        "before_excess_kurtosis":
            before.kurt(),
        "after_excess_kurtosis":
            after.kurt(),
        "before_IQR_outliers":
            before_outliers,
        "after_IQR_outliers":
            after_outliers
    })

transformation_validation = pd.DataFrame(
    validation_rows
)

transformation_validation.to_csv(
    TABLE_DIR / "19_before_after_validation.csv",
    index=False
)


# ============================================================
# 28. PLOT 1 - DATASET OVERVIEW
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10)
)

# Feature types
type_counts = pd.Series({
    "Numeric": len(numeric_cols),
    "Categorical": len(categorical_cols),
    "Datetime": len(datetime_cols)
})

type_counts.plot(
    kind="bar",
    ax=axes[0, 0]
)

axes[0, 0].set_title(
    "Dataset Feature Types"
)

axes[0, 0].set_ylabel(
    "Number of Columns"
)

axes[0, 0].tick_params(
    axis="x",
    rotation=0
)


# Missing values
top_missing = (
    missing
    .head(10)
    .sort_values("missing_percent")
)

axes[0, 1].barh(
    top_missing["column"],
    top_missing["missing_percent"]
)

axes[0, 1].set_title(
    "Top 10 Columns by Missing Percentage"
)

axes[0, 1].set_xlabel(
    "Missing Values (%)"
)


# Dataset size
axes[1, 0].bar(
    ["Rows", "Columns"],
    [df.shape[0], df.shape[1]]
)

axes[1, 0].set_title(
    "Dataset Dimensions"
)

axes[1, 0].set_ylabel(
    "Count"
)


# Data quality
quality_labels = [
    "Missing Cells",
    "Duplicates",
    "Constant Columns",
    "ID-like Columns"
]

quality_values = [
    int(df.isna().sum().sum()),
    int(df.duplicated().sum()),
    len(constant_cols),
    len(id_like_cols)
]

axes[1, 1].bar(
    quality_labels,
    quality_values
)

axes[1, 1].set_title(
    "Data Quality Indicators"
)

axes[1, 1].tick_params(
    axis="x",
    rotation=25
)

save_plot(
    "01_dataset_overview.png"
)


# ============================================================
# 29. PLOT 2 - NUMERICAL DISTRIBUTIONS
# ============================================================

distribution_cols = (
    variance_table
    .head(MAX_DISTRIBUTION_VARS)["column"]
    .tolist()
)

if distribution_cols:

    n_cols = 2
    n_rows = int(
        np.ceil(
            len(distribution_cols)
            / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows)
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax, col in zip(
        axes,
        distribution_cols
    ):

        values = safe_sample(
            df[col],
            MAX_PLOT_SAMPLE
        )

        sns.histplot(
            values,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Distribution: {col}"
        )

        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    for ax in axes[
        len(distribution_cols):
    ]:
        ax.axis("off")

    save_plot(
        "02_numerical_distributions.png"
    )


# ============================================================
# 30. PLOT 3 - BOX PLOTS / OUTLIERS
# ============================================================

box_cols = (
    variance_table
    .head(MAX_BOXPLOT_VARS)["column"]
    .tolist()
)

if box_cols:

    box_data = df[
        box_cols
    ].copy()

    box_data = box_data.apply(
        lambda x: (
            (x - x.median())
            / x.std()
            if x.std() != 0
            else x
        )
    )

    plt.figure(
        figsize=(15, 7)
    )

    sns.boxplot(
        data=box_data
    )

    plt.title(
        "Standardized Boxplots - Outlier Comparison"
    )

    plt.xlabel(
        "Numerical Variables"
    )

    plt.ylabel(
        "Standardized Value"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    save_plot(
        "03_outlier_boxplots.png"
    )


# ============================================================
# 31. PLOT 4 - CATEGORICAL DISTRIBUTIONS
# ============================================================

categorical_plot_cols = [
    col for col in categorical_cols
    if col not in id_like_cols
    and 2 <= df[col].nunique(dropna=True) <= 15
][:MAX_CATEGORICAL_VARS]

if categorical_plot_cols:

    n_cols = 2

    n_rows = int(
        np.ceil(
            len(categorical_plot_cols)
            / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 5 * n_rows)
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax, col in zip(
        axes,
        categorical_plot_cols
    ):

        counts = (
            df[col]
            .value_counts()
            .head(10)
            .sort_values()
        )

        ax.barh(
            counts.index.astype(str),
            counts.values
        )

        ax.set_title(
            f"Category Frequency: {col}"
        )

        ax.set_xlabel("Count")
        ax.set_ylabel(col)

    for ax in axes[
        len(categorical_plot_cols):
    ]:
        ax.axis("off")

    save_plot(
        "04_categorical_distributions.png"
    )


# ============================================================
# 32. PLOT 5 - PEARSON CORRELATION
# ============================================================

if len(correlation_candidates) >= 2:

    plt.figure(
        figsize=(13, 10)
    )

    sns.heatmap(
        pearson_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5
    )

    plt.title(
        "Pearson Correlation Matrix"
    )

    save_plot(
        "05_pearson_correlation.png"
    )


# ============================================================
# 33. PLOT 6 - SPEARMAN CORRELATION
# ============================================================

if len(correlation_candidates) >= 2:

    plt.figure(
        figsize=(13, 10)
    )

    sns.heatmap(
        spearman_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5
    )

    plt.title(
        "Spearman Rank Correlation Matrix"
    )

    save_plot(
        "06_spearman_correlation.png"
    )


# ============================================================
# 34. PLOT 7 - STRONGEST CORRELATION PAIRS
# ============================================================

if (
    "correlation_pairs" in locals()
    and not correlation_pairs.empty
):

    top_scatter_pairs = (
        correlation_pairs
        .head(4)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12)
    )

    axes = axes.flatten()

    for ax, (_, row) in zip(
        axes,
        top_scatter_pairs.iterrows()
    ):

        x = row["variable_1"]
        y = row["variable_2"]

        temp = df[
            [x, y]
        ].dropna()

        if len(temp) > MAX_PLOT_SAMPLE:

            temp = temp.sample(
                MAX_PLOT_SAMPLE,
                random_state=RANDOM_STATE
            )

        sns.scatterplot(
            data=temp,
            x=x,
            y=y,
            alpha=0.4,
            s=25,
            ax=ax
        )

        ax.set_title(
            f"{x} vs {y}\n"
            f"Pearson r = {row['pearson_r']:.2f}"
        )

    for ax in axes[
        len(top_scatter_pairs):
    ]:
        ax.axis("off")

    save_plot(
        "07_top_correlation_scatterplots.png"
    )


# ============================================================
# 35. PLOT 8 - SKEWNESS
# ============================================================

if not skewness.empty:

    skew_plot = skewness.head(10)

    plt.figure(
        figsize=(13, 7)
    )

    sns.barplot(
        data=skew_plot,
        x="skewness",
        y="column",
        hue="classification",
        dodge=False
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.axvline(
        1,
        linestyle=":"
    )

    plt.axvline(
        -1,
        linestyle=":"
    )

    plt.title(
        "Top 10 Variables by Absolute Skewness"
    )

    plt.xlabel(
        "Skewness"
    )

    plt.ylabel(
        "Variable"
    )

    save_plot(
        "08_skewness_analysis.png"
    )


# ============================================================
# 36. PLOT 9 - KURTOSIS
# ============================================================

if not kurtosis_table.empty:

    kurt_plot = kurtosis_table.head(10)

    plt.figure(
        figsize=(13, 7)
    )

    sns.barplot(
        data=kurt_plot,
        x="excess_kurtosis",
        y="column",
        hue="classification",
        dodge=False
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.axvline(
        3,
        linestyle=":"
    )

    plt.axvline(
        -3,
        linestyle=":"
    )

    plt.title(
        "Top 10 Variables by Absolute Excess Kurtosis"
    )

    plt.xlabel(
        "Excess Kurtosis"
    )

    plt.ylabel(
        "Variable"
    )

    save_plot(
        "09_excess_kurtosis_analysis.png"
    )


# ============================================================
# 37. PLOT 10 - Q-Q NORMALITY
# ============================================================

if normality_candidates:

    n_cols = 2

    n_rows = int(
        np.ceil(
            len(normality_candidates)
            / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 5 * n_rows)
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax, col in zip(
        axes,
        normality_candidates
    ):

        values = safe_sample(
            df[col],
            MAX_NORMALITY_SAMPLE
        )

        stats.probplot(
            values,
            dist="norm",
            plot=ax
        )

        ax.set_title(
            f"Q-Q Plot: {col}"
        )

    for ax in axes[
        len(normality_candidates):
    ]:
        ax.axis("off")

    save_plot(
        "10_qq_normality_plots.png"
    )


# ============================================================
# 38. PLOT 11 - GROUP MEANS + CONFIDENCE INTERVAL
# ============================================================

if (
    group_target is not None
    and group_candidates
):

    group_plot_col = group_candidates[0]

    temp = df[
        [group_plot_col, group_target]
    ].dropna()

    if not temp.empty:

        summary = (
            temp.groupby(
                group_plot_col,
                observed=True
            )[group_target]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        summary["sem"] = (
            summary["std"]
            / np.sqrt(summary["count"])
        )

        summary["ci95"] = (
            1.96 * summary["sem"]
        )

        plt.figure(
            figsize=(13, 7)
        )

        plt.bar(
            summary[
                group_plot_col
            ].astype(str),
            summary["mean"],
            yerr=summary["ci95"],
            capsize=5
        )

        plt.title(
            f"Mean {group_target} by {group_plot_col}"
        )

        plt.xlabel(
            group_plot_col
        )

        plt.ylabel(
            f"Mean {group_target}"
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        save_plot(
            "11_group_means_confidence_interval.png"
        )


# ============================================================
# 39. PLOT 12 - GROUP DISTRIBUTION
# ============================================================

if (
    group_target is not None
    and group_candidates
):

    group_plot_col = group_candidates[0]

    temp = df[
        [group_plot_col, group_target]
    ].dropna()

    if not temp.empty:

        if len(temp) > MAX_PLOT_SAMPLE:

            temp = temp.sample(
                MAX_PLOT_SAMPLE,
                random_state=RANDOM_STATE
            )

        plt.figure(
            figsize=(13, 7)
        )

        sns.boxplot(
            data=temp,
            x=group_plot_col,
            y=group_target
        )

        plt.title(
            f"{group_target} Distribution Across {group_plot_col}"
        )

        plt.xlabel(
            group_plot_col
        )

        plt.ylabel(
            group_target
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        save_plot(
            "12_group_distribution_boxplot.png"
        )


# ============================================================
# 40. PLOT 13 - STATISTICAL EFFECT SIZES
# ============================================================

if (
    "statistical_tests" in locals()
    and not statistical_tests.empty
):

    effect_plot = (
        statistical_tests[
            statistical_tests["effect_size"]
            .notna()
        ]
        .copy()
    )

    effect_plot["abs_effect"] = (
        effect_plot["effect_size"]
        .abs()
    )

    effect_plot = effect_plot.sort_values(
        "abs_effect",
        ascending=False
    ).head(15)

    if not effect_plot.empty:

        effect_plot["label"] = (
            effect_plot["variable_1"].astype(str)
            + " ↔ "
            + effect_plot["variable_2"].astype(str)
        )

        plt.figure(
            figsize=(14, 9)
        )

        sns.barplot(
            data=effect_plot,
            x="effect_size",
            y="label",
            hue="effect_magnitude",
            dodge=False
        )

        plt.axvline(
            0,
            linestyle="--"
        )

        plt.title(
            "Top Statistical Effect Sizes"
        )

        plt.xlabel(
            "Signed Effect Size"
        )

        plt.ylabel(
            "Variable Relationship"
        )

        save_plot(
            "13_statistical_effect_sizes.png"
        )


# ============================================================
# 41. PLOT 14 - VIF
# ============================================================

if (
    "vif_results" in locals()
    and not vif_results.empty
):

    plt.figure(
        figsize=(13, 8)
    )

    sns.barplot(
        data=vif_results,
        x="VIF",
        y="feature",
        hue="interpretation",
        dodge=False
    )

    plt.axvline(
        5,
        linestyle="--",
        label="VIF = 5"
    )

    plt.axvline(
        10,
        linestyle=":",
        label="VIF = 10"
    )

    plt.title(
        "Variance Inflation Factor - Multicollinearity"
    )

    plt.xlabel(
        "VIF"
    )

    plt.ylabel(
        "Feature"
    )

    plt.legend()

    save_plot(
        "14_vif_multicollinearity.png"
    )


# ============================================================
# 42. PLOT 15 - BEFORE VS AFTER SKEWNESS
# ============================================================

if not transformation_validation.empty:

    plot_validation = (
        transformation_validation
        .head(15)
        .copy()
    )

    x = np.arange(
        len(plot_validation)
    )

    width = 0.35

    plt.figure(
        figsize=(14, 8)
    )

    plt.bar(
        x - width / 2,
        plot_validation[
            "before_skewness"
        ],
        width,
        label="Before"
    )

    plt.bar(
        x + width / 2,
        plot_validation[
            "after_skewness"
        ],
        width,
        label="After Yeo-Johnson"
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.axhline(
        1,
        linestyle=":"
    )

    plt.axhline(
        -1,
        linestyle=":"
    )

    plt.xticks(
        x,
        plot_validation["column"],
        rotation=45,
        ha="right"
    )

    plt.title(
        "Skewness Before vs After Yeo-Johnson Transformation"
    )

    plt.ylabel(
        "Skewness"
    )

    plt.legend()

    save_plot(
        "15_before_after_skewness.png"
    )


# ============================================================
# 43. FINAL REPORT
# ============================================================

report_lines = []

report_lines.append(
    "OLIST DATASET - EDA + STATISTICAL TESTING REPORT"
)

report_lines.append("=" * 70)

report_lines.append(
    f"Dataset shape: {df.shape}"
)

report_lines.append(
    f"Numeric columns: {len(numeric_cols)}"
)

report_lines.append(
    f"Categorical columns: {len(categorical_cols)}"
)

report_lines.append(
    f"Datetime columns: {len(datetime_cols)}"
)

report_lines.append(
    f"Duplicate rows: {df.duplicated().sum()}"
)

report_lines.append(
    f"Missing cells: {int(df.isna().sum().sum())}"
)

report_lines.append(
    f"Constant columns: {len(constant_cols)}"
)

report_lines.append(
    f"ID-like columns: {len(id_like_cols)}"
)

report_lines.append("")

report_lines.append(
    "SKEWNESS SUMMARY"
)

report_lines.append("-" * 70)

report_lines.append(
    f"Highly skewed variables: "
    f"{len(high_skew_cols)}"
)

report_lines.append("")

report_lines.append(
    "KURTOSIS SUMMARY"
)

report_lines.append("-" * 70)

if not kurtosis_table.empty:

    high_kurtosis_count = int(
        kurtosis_table["high_kurtosis"].sum()
    )

else:

    high_kurtosis_count = 0

report_lines.append(
    f"Variables with |excess kurtosis| > 3: "
    f"{high_kurtosis_count}"
)

report_lines.append("")

report_lines.append(
    "NORMALITY SUMMARY"
)

report_lines.append("-" * 70)

if not normality_results.empty:

    normality_results["both_tests_pass"] = (
        (normality_results["shapiro_p_value"] >= 0.05)
        &
        (normality_results["dagostino_p_value"] >= 0.05)
    )

    passed_normality = int(
        normality_results[
            "both_tests_pass"
        ].sum()
    )

    report_lines.append(
        f"Variables passing both normality tests: "
        f"{passed_normality} / "
        f"{len(normality_results)}"
    )

report_lines.append("")

report_lines.append(
    "MULTICOLLINEARITY SUMMARY"
)

report_lines.append("-" * 70)

if "vif_results" in locals():

    high_vif = int(
        (
            vif_results["VIF"] >= 10
        ).sum()
    )

    moderate_vif = int(
        (
            (vif_results["VIF"] >= 5)
            &
            (vif_results["VIF"] < 10)
        ).sum()
    )

    report_lines.append(
        f"High VIF >= 10: {high_vif}"
    )

    report_lines.append(
        f"Moderate VIF 5-10: {moderate_vif}"
    )

report_lines.append("")

report_lines.append(
    "STATISTICAL TESTING SUMMARY"
)

report_lines.append("-" * 70)

if not statistical_tests.empty:

    raw_sig = int(
        statistical_tests[
            "significant_raw"
        ].sum()
    )

    fdr_sig = int(
        statistical_tests[
            "significant_fdr"
        ].sum()
    )

    report_lines.append(
        f"Raw significant results (p < 0.05): "
        f"{raw_sig}"
    )

    report_lines.append(
        f"FDR significant results: "
        f"{fdr_sig}"
    )

report_lines.append("")

report_lines.append(
    "TRANSFORMATION SUMMARY"
)

report_lines.append("-" * 70)

if not transformation_validation.empty:

    improved = int(
        (
            transformation_validation[
                "after_skewness"
            ].abs()
            <
            transformation_validation[
                "before_skewness"
            ].abs()
        ).sum()
    )

    report_lines.append(
        f"Variables improved after Yeo-Johnson: "
        f"{improved} / "
        f"{len(transformation_validation)}"
    )

report_lines.append("")

report_lines.append(
    "IMPORTANT INTERPRETATION:"
)

report_lines.append(
    "Correlation does not prove causation."
)

report_lines.append(
    "A statistically significant result does not automatically mean "
    "the effect is practically important."
)

report_lines.append(
    "Large datasets can produce very small p-values, so effect sizes "
    "should be considered together with p-values."
)

report_lines.append(
    "Yeo-Johnson transformation addresses distributional skewness; "
    "it should not be described as an automatic solution to multicollinearity."
)

report_path = (
    REPORT_DIR
    / "olist_eda_statistical_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ============================================================
# 44. FINAL CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    f"\nPlots saved in:\n{PLOT_DIR}"
)

print(
    f"\nTables saved in:\n{TABLE_DIR}"
)

print(
    f"\nReport saved in:\n{REPORT_DIR}"
)

print("\nGenerated plot files:")

for plot_file in sorted(
    PLOT_DIR.glob("*.png")
):
    print(
        f"  - {plot_file.name}"
    )

print("\nGenerated table files:")

for table_file in sorted(
    TABLE_DIR.glob("*.csv")
):
    print(
        f"  - {table_file.name}"
    )

print("\n" + "=" * 80)
print("GITHUB COMMANDS")
print("=" * 80)

print("""
git status
git add .
git commit -m "Improve OLIST EDA and statistical analysis"
git push origin main
""")

print("=" * 80)
print("DONE")
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