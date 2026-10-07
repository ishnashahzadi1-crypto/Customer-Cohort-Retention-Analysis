# ============================================================
# OLIST DATASET - COMPLETE EDA + STATISTICAL TESTING
# Professional, Fast, Clean and Reproducible
# ============================================================

"""
This script performs:

1. Dataset overview
2. Data quality analysis
3. Missing-value analysis
4. Duplicate analysis
5. Infinite-value analysis
6. Constant / near-zero variance analysis
7. ID-like column detection
8. Descriptive statistics
9. Variance / standard deviation / IQR analysis
10. IQR and Z-score outlier analysis
11. Skewness analysis
12. Yeo-Johnson transformation for highly skewed variables
13. Kurtosis analysis
14. Normality analysis
15. Pearson and Spearman correlations
16. Mean-grouping / dependence analysis
17. Independent t-test / Mann-Whitney U
18. ANOVA / Kruskal-Wallis
19. Levene / Bartlett variance tests
20. Chi-square + Cramer's V
21. Benjamini-Hochberg FDR correction
22. VIF multicollinearity analysis
23. Before-vs-after validation
24. Professional consolidated visualizations
25. CSV output tables
26. Final report
27. Git commands for GitHub
"""

# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import warnings
warnings.filterwarnings("ignore")

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
    pearsonr,
    spearmanr,
    chi2_contingency
)

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Maximum number of variables shown in consolidated plots
MAX_DIST_VARS = 8
MAX_BOXPLOT_VARS = 10
MAX_CAT_VARS = 6
MAX_CORR_VARS = 12
MAX_SKEW_VARS = 10
MAX_KURTOSIS_VARS = 10
MAX_QQ_VARS = 6

# Statistical-test sample size
# Large datasets do not need every row for expensive tests.
MAX_TEST_SAMPLE = 5000

# Normality sample size
MAX_NORMALITY_SAMPLE = 5000

# Outlier Z-score threshold
Z_THRESHOLD = 3.0

# Near-zero variance threshold
NEAR_ZERO_VARIANCE_RATIO = 0.01


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for folder in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. FIND DATASET AUTOMATICALLY
# ============================================================

def find_dataset():
    """
    Searches for CSV files in:
    1. project root
    2. data/
    """

    possible_names = [
        "olist_6_files_merged.csv",
        "olist_6_files_merged(1).csv",
        "olist.csv"
    ]

    # First try known filenames
    for name in possible_names:
        root_file = PROJECT_ROOT / name
        data_file = DATA_DIR / name

        if root_file.exists():
            return root_file

        if data_file.exists():
            return data_file

    # Otherwise search for CSV files
    root_csvs = list(PROJECT_ROOT.glob("*.csv"))

    if root_csvs:
        return root_csvs[0]

    data_csvs = list(DATA_DIR.glob("*.csv"))

    if data_csvs:
        return data_csvs[0]

    raise FileNotFoundError(
        "\nNo CSV dataset was found.\n"
        "Put your CSV in the project root or inside the data/ folder."
    )


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("=" * 60)
print("OLIST DATASET - COMPLETE EDA + STATISTICAL TESTING")
print("=" * 60)

DATA_PATH = find_dataset()

print(f"\nDataset file: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Dataset loaded successfully.")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")


# ============================================================
# 5. BASIC CLEANING
# ============================================================

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Convert obvious datetime columns
datetime_columns = []

for col in df.columns:

    if df[col].dtype == "object":

        sample = df[col].dropna().astype(str).head(1000)

        if len(sample) > 0:

            parsed = pd.to_datetime(
                sample,
                errors="coerce"
            )

            conversion_rate = parsed.notna().mean()

            if conversion_rate >= 0.80:

                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                datetime_columns.append(col)


# ============================================================
# 6. COLUMN TYPE IDENTIFICATION
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

datetime_cols = df.select_dtypes(
    include=["datetime64[ns]", "datetime64[ns, UTC]"]
).columns.tolist()


print("\n" + "=" * 60)
print("COLUMN TYPES")
print("=" * 60)

print(f"Numerical columns : {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Datetime columns   : {len(datetime_cols)}")


# ============================================================
# 7. DATASET INFORMATION
# ============================================================

dataset_info = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Numerical Features",
        "Categorical Features",
        "Datetime Features",
        "Missing Cells",
        "Duplicate Rows"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        len(numeric_cols),
        len(categorical_cols),
        len(datetime_cols),
        int(df.isna().sum().sum()),
        int(df.duplicated().sum())
    ]
})

dataset_info.to_csv(
    TABLES_DIR / "dataset_info.csv",
    index=False
)


# ============================================================
# 8. MISSING VALUE ANALYSIS
# ============================================================

missing_count = df.isna().sum()

missing_pct = (
    missing_count / len(df) * 100
).round(3)

missing_table = pd.DataFrame({
    "column": df.columns,
    "missing_count": missing_count.values,
    "missing_percentage": missing_pct.values
})

missing_table = missing_table.sort_values(
    "missing_percentage",
    ascending=False
)

missing_table.to_csv(
    TABLES_DIR / "missing_values.csv",
    index=False
)


# ============================================================
# 9. DUPLICATE ANALYSIS
# ============================================================

duplicate_count = int(df.duplicated().sum())

duplicate_summary = pd.DataFrame({
    "metric": [
        "duplicate_rows",
        "duplicate_percentage"
    ],
    "value": [
        duplicate_count,
        round(duplicate_count / len(df) * 100, 3)
    ]
})

duplicate_summary.to_csv(
    TABLES_DIR / "duplicates.csv",
    index=False
)


# ============================================================
# 10. UNIQUE VALUE ANALYSIS
# ============================================================

unique_table = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "unique_values": [
        df[col].nunique(dropna=False)
        for col in df.columns
    ],
    "unique_percentage": [
        round(
            df[col].nunique(dropna=False) / len(df) * 100,
            3
        )
        for col in df.columns
    ]
})

unique_table.to_csv(
    TABLES_DIR / "unique_value_counts.csv",
    index=False
)


# ============================================================
# 11. ID-LIKE COLUMN DETECTION
# ============================================================

id_like_cols = []

for col in df.columns:

    unique_ratio = (
        df[col].nunique(dropna=False) / len(df)
    )

    name_lower = col.lower()

    name_hint = (
        "id" in name_lower
        or name_lower.endswith("_key")
        or "uuid" in name_lower
    )

    if unique_ratio >= 0.95 or name_hint:
        id_like_cols.append(col)

id_like_table = pd.DataFrame({
    "id_like_column": id_like_cols
})

id_like_table.to_csv(
    TABLES_DIR / "id_like_columns.csv",
    index=False
)


# ============================================================
# 12. CONSTANT COLUMNS
# ============================================================

constant_cols = [
    col
    for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

constant_table = pd.DataFrame({
    "constant_column": constant_cols
})

constant_table.to_csv(
    TABLES_DIR / "constant_columns.csv",
    index=False
)


# ============================================================
# 13. NUMERICAL DATAFRAME
# ============================================================

# Remove constant numeric columns from numerical analysis
analysis_numeric_cols = [
    col
    for col in numeric_cols
    if col not in constant_cols
]

numeric_df = df[analysis_numeric_cols].copy()


# ============================================================
# 14. VARIANCE / DESCRIPTIVE STATISTICS
# ============================================================

descriptive = numeric_df.describe(
    percentiles=[0.25, 0.50, 0.75]
).T

descriptive["variance"] = numeric_df.var(
    ddof=1
)

descriptive["IQR"] = (
    numeric_df.quantile(0.75)
    - numeric_df.quantile(0.25)
)

descriptive["missing_count"] = numeric_df.isna().sum()

descriptive["missing_percentage"] = (
    numeric_df.isna().mean() * 100
)

descriptive = descriptive.reset_index()

descriptive = descriptive.rename(
    columns={"index": "variable"}
)

descriptive.to_csv(
    TABLES_DIR / "descriptive_statistics.csv",
    index=False
)


# ============================================================
# 15. NEAR-ZERO VARIANCE ANALYSIS
# ============================================================

near_zero_variance = []

for col in analysis_numeric_cols:

    series = numeric_df[col].dropna()

    if len(series) == 0:
        continue

    variance = series.var()

    if variance == 0:
        status = "Zero variance"

    else:
        most_common_frequency = (
            series.value_counts(normalize=True)
            .iloc[0]
        )

        if most_common_frequency >= (
            1 - NEAR_ZERO_VARIANCE_RATIO
        ):
            status = "Near-zero variance"
        else:
            status = "Normal variance"

    near_zero_variance.append({
        "variable": col,
        "variance": variance,
        "status": status
    })

variance_status = pd.DataFrame(
    near_zero_variance
)

variance_status.to_csv(
    TABLES_DIR / "variance_analysis.csv",
    index=False
)


# ============================================================
# 16. OUTLIER ANALYSIS - IQR + Z-SCORE
# ============================================================

outlier_results = []

for col in analysis_numeric_cols:

    series = numeric_df[col].dropna()

    if len(series) < 10:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    iqr_mask = (
        (series < lower_bound)
        | (series > upper_bound)
    )

    iqr_count = int(iqr_mask.sum())

    # Z-score
    mean_value = series.mean()
    std_value = series.std()

    if std_value > 0:

        z_scores = (
            (series - mean_value)
            / std_value
        )

        z_count = int(
            (np.abs(z_scores) > Z_THRESHOLD).sum()
        )

    else:
        z_count = 0

    outlier_results.append({
        "variable": col,
        "IQR_outliers": iqr_count,
        "IQR_outlier_percentage": round(
            iqr_count / len(series) * 100,
            3
        ),
        "Z_score_outliers": z_count,
        "Z_score_outlier_percentage": round(
            z_count / len(series) * 100,
            3
        ),
        "lower_IQR_bound": lower_bound,
        "upper_IQR_bound": upper_bound
    })

outlier_summary = pd.DataFrame(
    outlier_results
)

outlier_summary = outlier_summary.sort_values(
    "IQR_outlier_percentage",
    ascending=False
)

outlier_summary.to_csv(
    TABLES_DIR / "outlier_summary.csv",
    index=False
)


# ============================================================
# 17. SKEWNESS ANALYSIS
# ============================================================

skew_results = []

for col in analysis_numeric_cols:

    series = numeric_df[col].dropna()

    if len(series) < 3:
        continue

    skew_value = series.skew()

    abs_skew = abs(skew_value)

    if abs_skew <= 0.5:
        classification = "Approximately symmetric"

    elif abs_skew <= 1:
        classification = "Moderately skewed"

    else:
        classification = "Highly skewed"

    skew_results.append({
        "variable": col,
        "skewness_before": skew_value,
        "absolute_skewness": abs_skew,
        "classification": classification
    })

skewness_analysis = pd.DataFrame(
    skew_results
).sort_values(
    "absolute_skewness",
    ascending=False
)

skewness_analysis.to_csv(
    TABLES_DIR / "skewness_analysis.csv",
    index=False
)


# ============================================================
# 18. YEO-JOHNSON TRANSFORMATION
# ============================================================

"""
We do NOT transform every numerical variable.

Only highly skewed variables are considered.

Yeo-Johnson is selected because it can handle:
- positive values
- zero values
- negative values

Variables with insufficient variation are skipped.
"""

highly_skewed_cols = skewness_analysis.loc[
    skewness_analysis["absolute_skewness"] > 1,
    "variable"
].tolist()

# Limit transformations for performance
transform_cols = highly_skewed_cols[:15]

transformed_numeric_df = numeric_df.copy()

transformation_results = []

for col in transform_cols:

    series = numeric_df[col]

    valid_mask = series.notna()

    valid_values = series.loc[valid_mask].values.reshape(-1, 1)

    if len(valid_values) < 20:
        continue

    if np.nanstd(valid_values) == 0:
        continue

    try:

        transformer = PowerTransformer(
            method="yeo-johnson",
            standardize=False
        )

        transformed_values = transformer.fit_transform(
            valid_values
        ).ravel()

        transformed_series = pd.Series(
            index=series.index,
            dtype=float
        )

        transformed_series.loc[
            valid_mask
        ] = transformed_values

        transformed_numeric_df[col] = transformed_series

        before_skew = series.dropna().skew()
        after_skew = transformed_series.dropna().skew()

        transformation_results.append({
            "variable": col,
            "transformation": "Yeo-Johnson",
            "skewness_before": before_skew,
            "skewness_after": after_skew,
            "absolute_skewness_before": abs(before_skew),
            "absolute_skewness_after": abs(after_skew)
        })

    except Exception as error:

        transformation_results.append({
            "variable": col,
            "transformation": "Skipped",
            "skewness_before": series.dropna().skew(),
            "skewness_after": np.nan,
            "absolute_skewness_before": abs(
                series.dropna().skew()
            ),
            "absolute_skewness_after": np.nan
        })

transformation_table = pd.DataFrame(
    transformation_results
)

transformation_table.to_csv(
    TABLES_DIR / "skewness_before_after.csv",
    index=False
)


# ============================================================
# 19. KURTOSIS ANALYSIS
# ============================================================

kurtosis_results = []

for col in analysis_numeric_cols:

    series = numeric_df[col].dropna()

    if len(series) < 4:
        continue

    kurtosis_value = series.kurt()

    if kurtosis_value > 3:
        classification = "High / heavy-tailed"

    elif kurtosis_value > 1:
        classification = "Moderately high"

    elif kurtosis_value < -1:
        classification = "Light-tailed"

    else:
        classification = "Typical range"

    kurtosis_results.append({
        "variable": col,
        "excess_kurtosis_before": kurtosis_value,
        "absolute_excess_kurtosis": abs(kurtosis_value),
        "classification": classification
    })

kurtosis_analysis = pd.DataFrame(
    kurtosis_results
).sort_values(
    "absolute_excess_kurtosis",
    ascending=False
)

kurtosis_analysis.to_csv(
    TABLES_DIR / "kurtosis_analysis.csv",
    index=False
)


# ============================================================
# 20. OUTLIER VALIDATION AFTER TRANSFORMATION
# ============================================================

after_outlier_results = []

for col in transform_cols:

    series = transformed_numeric_df[col].dropna()

    if len(series) < 10:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = int(
        ((series < lower) | (series > upper)).sum()
    )

    after_outlier_results.append({
        "variable": col,
        "IQR_outliers_after": count,
        "IQR_outlier_percentage_after": round(
            count / len(series) * 100,
            3
        )
    })

after_outlier_table = pd.DataFrame(
    after_outlier_results
)

if not after_outlier_table.empty:

    before_part = outlier_summary[
        [
            "variable",
            "IQR_outlier_percentage"
        ]
    ].rename(
        columns={
            "IQR_outlier_percentage":
                "IQR_outlier_percentage_before"
        }
    )

    outlier_before_after = before_part.merge(
        after_outlier_table,
        on="variable",
        how="inner"
    )

else:

    outlier_before_after = pd.DataFrame()

outlier_before_after.to_csv(
    TABLES_DIR / "outlier_before_after.csv",
    index=False
)


# ============================================================
# 21. NORMALITY TESTING
# ============================================================

"""
Normality testing strategy:

- Shapiro-Wilk is limited to a sample of 5,000.
- D'Agostino-Pearson is also sampled.
- We report skewness and kurtosis.
- We do not interpret p-values alone.
- Large datasets can produce extremely small p-values
  even for practically small deviations from normality.
"""

normality_candidates = []

# Select variables with reasonable variation
for col in analysis_numeric_cols:

    series = numeric_df[col].dropna()

    if len(series) >= 20 and series.nunique() > 5:
        normality_candidates.append(col)

# Select variables with strongest variation
variance_rank = (
    numeric_df[normality_candidates]
    .var()
    .sort_values(ascending=False)
)

normality_cols = variance_rank.head(
    MAX_QQ_VARS
).index.tolist()

normality_results = []

for col in normality_cols:

    series = numeric_df[col].dropna()

    if len(series) > MAX_NORMALITY_SAMPLE:

        sample = series.sample(
            MAX_NORMALITY_SAMPLE,
            random_state=RANDOM_STATE
        )

    else:

        sample = series

    # Shapiro-Wilk
    try:

        shapiro_stat, shapiro_p = shapiro(
            sample
        )

    except Exception:

        shapiro_stat = np.nan
        shapiro_p = np.nan

    # D'Agostino-Pearson
    try:

        normaltest_stat, normaltest_p = normaltest(
            sample
        )

    except Exception:

        normaltest_stat = np.nan
        normaltest_p = np.nan

    skew_value = sample.skew()
    kurt_value = sample.kurt()

    # Practical interpretation
    if (
        abs(skew_value) <= 0.5
        and abs(kurt_value) <= 1
    ):
        practical_normality = (
            "Approximately normal by shape"
        )

    elif (
        abs(skew_value) <= 1
        and abs(kurt_value) <= 2
    ):
        practical_normality = (
            "Moderate deviation from normality"
        )

    else:
        practical_normality = (
            "Strong deviation from normality"
        )

    normality_results.append({
        "variable": col,
        "sample_size": len(sample),
        "shapiro_statistic": shapiro_stat,
        "shapiro_p_value": shapiro_p,
        "dagostino_statistic": normaltest_stat,
        "dagostino_p_value": normaltest_p,
        "skewness": skew_value,
        "excess_kurtosis": kurt_value,
        "practical_normality": practical_normality
    })

normality_tests = pd.DataFrame(
    normality_results
)

normality_tests.to_csv(
    TABLES_DIR / "normality_tests.csv",
    index=False
)


# ============================================================
# 22. CORRELATION ANALYSIS
# ============================================================

# Select variables with highest variance
corr_candidate_cols = (
    numeric_df.var()
    .sort_values(ascending=False)
    .head(MAX_CORR_VARS)
    .index
    .tolist()
)

corr_df = numeric_df[
    corr_candidate_cols
].copy()

pearson_matrix = corr_df.corr(
    method="pearson"
)

spearman_matrix = corr_df.corr(
    method="spearman"
)

pearson_matrix.to_csv(
    TABLES_DIR / "pearson_correlation_matrix.csv"
)

spearman_matrix.to_csv(
    TABLES_DIR / "spearman_correlation_matrix.csv"
)


# ============================================================
# 23. PAIRWISE CORRELATION SUMMARY
# ============================================================

correlation_results = []

for i, col1 in enumerate(corr_candidate_cols):

    for col2 in corr_candidate_cols[i + 1:]:

        pair = corr_df[
            [col1, col2]
        ].dropna()

        if len(pair) < 10:
            continue

        try:

            pearson_r, pearson_p = pearsonr(
                pair[col1],
                pair[col2]
            )

        except Exception:

            pearson_r = np.nan
            pearson_p = np.nan

        try:

            spearman_r, spearman_p = spearmanr(
                pair[col1],
                pair[col2]
            )

        except Exception:

            spearman_r = np.nan
            spearman_p = np.nan

        correlation_results.append({
            "variable_1": col1,
            "variable_2": col2,
            "pearson_r": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p_value": spearman_p,
            "absolute_pearson_r": abs(pearson_r)
        })

correlation_table = pd.DataFrame(
    correlation_results
).sort_values(
    "absolute_pearson_r",
    ascending=False
)

correlation_table.to_csv(
    TABLES_DIR / "correlation_analysis.csv",
    index=False
)


# ============================================================
# 24. HELPER FUNCTIONS FOR EFFECT SIZES
# ============================================================

def cohens_d(group1, group2):
    """
    Cohen's d for two independent groups.
    """

    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    group1 = group1[np.isfinite(group1)]
    group2 = group2[np.isfinite(group2)]

    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        return np.nan

    variance1 = np.var(group1, ddof=1)
    variance2 = np.var(group2, ddof=1)

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * variance1
            + (n2 - 1) * variance2
        )
        / (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (
        np.mean(group1) - np.mean(group2)
    ) / pooled_sd


def eta_squared_from_groups(groups):
    """
    Eta-squared effect size for ANOVA.
    """

    clean_groups = [
        np.asarray(g, dtype=float)
        for g in groups
        if len(g) > 0
    ]

    if len(clean_groups) < 2:
        return np.nan

    all_values = np.concatenate(clean_groups)

    grand_mean = np.mean(all_values)

    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in clean_groups
    )

    ss_total = np.sum(
        (all_values - grand_mean) ** 2
    )

    if ss_total == 0:
        return np.nan

    return ss_between / ss_total


def cramers_v(contingency_table):
    """
    Bias-corrected Cramer's V.
    """

    chi2_value = chi2_contingency(
        contingency_table
    )[0]

    n = contingency_table.to_numpy().sum()

    if n == 0:
        return np.nan

    phi2 = chi2_value / n

    rows, cols = contingency_table.shape

    if n <= 1:
        return np.nan

    phi2corr = max(
        0,
        phi2 - (
            (cols - 1) * (rows - 1)
        ) / (n - 1)
    )

    rcorr = (
        rows
        - ((rows - 1) ** 2 / (n - 1))
    )

    kcorr = (
        cols
        - ((cols - 1) ** 2 / (n - 1))
    )

    denominator = min(
        kcorr - 1,
        rcorr - 1
    )

    if denominator <= 0:
        return np.nan

    return np.sqrt(
        phi2corr / denominator
    )


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.

    Returns adjusted p-values in the original order.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan
    )

    valid_mask = np.isfinite(p_values)

    valid_p = p_values[valid_mask]

    if len(valid_p) == 0:
        return adjusted

    order = np.argsort(valid_p)

    sorted_p = valid_p[order]

    m = len(sorted_p)

    adjusted_sorted = (
        sorted_p
        * m
        / np.arange(1, m + 1)
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1
    )

    valid_adjusted = np.empty(m)

    valid_adjusted[order] = adjusted_sorted

    adjusted[valid_mask] = valid_adjusted

    return adjusted


# ============================================================
# 25. GROUPING CATEGORICAL VARIABLES
# ============================================================

# Avoid ID-like and datetime columns
group_categorical_candidates = []

for col in categorical_cols:

    if col in id_like_cols:
        continue

    unique_count = df[col].nunique(dropna=True)

    if 2 <= unique_count <= 15:

        group_categorical_candidates.append(col)


# Also allow low-cardinality numeric categorical variables
for col in analysis_numeric_cols:

    unique_count = df[col].nunique(dropna=True)

    if (
        2 <= unique_count <= 15
        and col not in id_like_cols
    ):
        group_categorical_candidates.append(col)

# Remove duplicates while preserving order
group_categorical_candidates = list(
    dict.fromkeys(
        group_categorical_candidates
    )
)


# ============================================================
# 26. SELECT IMPORTANT NUMERICAL TARGETS
# ============================================================

group_numeric_candidates = [
    col
    for col in analysis_numeric_cols
    if df[col].nunique(dropna=True) > 5
]

group_numeric_candidates = (
    numeric_df[group_numeric_candidates]
    .var()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)


# ============================================================
# 27. GROUP-WISE MEAN / MEDIAN / STD
# ============================================================

group_results = []

# Limit the number of combinations
selected_group_categories = (
    group_categorical_candidates[:5]
)

selected_group_numerics = (
    group_numeric_candidates[:4]
)

for cat_col in selected_group_categories:

    for num_col in selected_group_numerics:

        temp = df[
            [cat_col, num_col]
        ].dropna()

        if temp.empty:
            continue

        grouped = temp.groupby(
            cat_col
        )[num_col].agg(
            ["count", "mean", "median", "std"]
        ).reset_index()

        for _, row in grouped.iterrows():

            group_results.append({
                "categorical_variable": cat_col,
                "category": row[cat_col],
                "numerical_variable": num_col,
                "count": row["count"],
                "mean": row["mean"],
                "median": row["median"],
                "std": row["std"]
            })

group_mean_analysis = pd.DataFrame(
    group_results
)

group_mean_analysis.to_csv(
    TABLES_DIR / "group_mean_analysis.csv",
    index=False
)


# ============================================================
# 28. STATISTICAL TESTING
# ============================================================

statistical_results = []


def add_result(
    test_name,
    variable_1,
    variable_2,
    statistic,
    p_value,
    effect_size,
    interpretation
):

    statistical_results.append({
        "test": test_name,
        "variable_1": variable_1,
        "variable_2": variable_2,
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": effect_size,
        "interpretation": interpretation
    })


# ============================================================
# 29. NUMERICAL VS NUMERICAL
# ============================================================

# Only the strongest correlation pairs
correlation_pairs_for_testing = (
    correlation_table.head(25)
)

for _, row in correlation_pairs_for_testing.iterrows():

    col1 = row["variable_1"]
    col2 = row["variable_2"]

    pair = corr_df[
        [col1, col2]
    ].dropna()

    if len(pair) < 20:
        continue

    if len(pair) > MAX_TEST_SAMPLE:

        pair = pair.sample(
            MAX_TEST_SAMPLE,
            random_state=RANDOM_STATE
        )

    try:

        pearson_r, pearson_p = pearsonr(
            pair[col1],
            pair[col2]
        )

        add_result(
            "Pearson correlation",
            col1,
            col2,
            pearson_r,
            pearson_p,
            abs(pearson_r),
            "Linear association"
        )

    except Exception:
        pass

    try:

        spearman_r, spearman_p = spearmanr(
            pair[col1],
            pair[col2]
        )

        add_result(
            "Spearman correlation",
            col1,
            col2,
            spearman_r,
            spearman_p,
            abs(spearman_r),
            "Monotonic association"
        )

    except Exception:
        pass


# ============================================================
# 30. TWO-GROUP + MULTI-GROUP TESTING
# ============================================================

for cat_col in selected_group_categories:

    for num_col in selected_group_numerics:

        temp = df[
            [cat_col, num_col]
        ].dropna()

        if temp.empty:
            continue

        groups = [
            group[num_col].values
            for _, group in temp.groupby(cat_col)
        ]

        group_labels = [
            label
            for label, _ in temp.groupby(cat_col)
        ]

        # Keep groups with enough observations
        filtered = [
            (label, values)
            for label, values in zip(
                group_labels,
                groups
            )
            if len(values) >= 10
        ]

        if len(filtered) < 2:
            continue

        group_labels = [
            x[0] for x in filtered
        ]

        groups = [
            x[1] for x in filtered
        ]

        # ----------------------------------------------------
        # TWO GROUPS
        # ----------------------------------------------------

        if len(groups) == 2:

            g1 = groups[0]
            g2 = groups[1]

            # Sample for speed
            if len(g1) > MAX_TEST_SAMPLE:
                g1 = pd.Series(g1).sample(
                    MAX_TEST_SAMPLE,
                    random_state=RANDOM_STATE
                ).values

            if len(g2) > MAX_TEST_SAMPLE:
                g2 = pd.Series(g2).sample(
                    MAX_TEST_SAMPLE,
                    random_state=RANDOM_STATE
                ).values

            # Levene
            try:

                lev_stat, lev_p = levene(
                    g1,
                    g2,
                    center="median"
                )

            except Exception:

                lev_stat = np.nan
                lev_p = np.nan

            add_result(
                "Levene variance test",
                cat_col,
                num_col,
                lev_stat,
                lev_p,
                np.nan,
                "Tests equality of variance"
            )

            # Normality evidence
            normality_reasonable = True

            for group_values in [g1, g2]:

                sample_for_normality = group_values

                if len(sample_for_normality) > 5000:

                    sample_for_normality = (
                        pd.Series(group_values)
                        .sample(
                            5000,
                            random_state=RANDOM_STATE
                        )
                        .values
                    )

                if len(sample_for_normality) >= 20:

                    try:

                        _, p_norm = normaltest(
                            sample_for_normality
                        )

                        if p_norm < 0.01:
                            normality_reasonable = False

                    except Exception:
                        pass

            # ------------------------------------------------
            # PARAMETRIC TEST
            # ------------------------------------------------

            if normality_reasonable and (
                np.isnan(lev_p)
                or lev_p >= 0.05
            ):

                try:

                    stat_value, p_value = ttest_ind(
                        g1,
                        g2,
                        equal_var=True
                    )

                    d = cohens_d(
                        g1,
                        g2
                    )

                    add_result(
                        "Independent t-test",
                        f"{cat_col}: {group_labels[0]}",
                        f"{cat_col}: {group_labels[1]} -> {num_col}",
                        stat_value,
                        p_value,
                        d,
                        "Parametric comparison of two groups"
                    )

                except Exception:
                    pass

            else:

                # ------------------------------------------------
                # NON-PARAMETRIC TEST
                # ------------------------------------------------

                try:

                    stat_value, p_value = mannwhitneyu(
                        g1,
                        g2,
                        alternative="two-sided"
                    )

                    # Rank-biserial effect size
                    n1 = len(g1)
                    n2 = len(g2)

                    rank_biserial = (
                        1
                        - (
                            2 * stat_value
                            / (n1 * n2)
                        )
                    )

                    add_result(
                        "Mann-Whitney U",
                        f"{cat_col}: {group_labels[0]}",
                        f"{cat_col}: {group_labels[1]} -> {num_col}",
                        stat_value,
                        p_value,
                        abs(rank_biserial),
                        "Non-parametric comparison of two groups"
                    )

                except Exception:
                    pass

        # ====================================================
        # MORE THAN TWO GROUPS
        # ====================================================

        elif len(groups) > 2:

            # ------------------------------------------------
            # Levene
            # ------------------------------------------------

            try:

                lev_stat, lev_p = levene(
                    *groups,
                    center="median"
                )

                add_result(
                    "Levene variance test",
                    cat_col,
                    num_col,
                    lev_stat,
                    lev_p,
                    np.nan,
                    "Tests equality of group variances"
                )

            except Exception:

                lev_p = np.nan

            # ------------------------------------------------
            # Normality assessment
            # ------------------------------------------------

            normality_reasonable = True

            for group_values in groups:

                if len(group_values) < 20:
                    normality_reasonable = False
                    break

                sample_values = group_values

                if len(sample_values) > 5000:

                    sample_values = (
                        pd.Series(sample_values)
                        .sample(
                            5000,
                            random_state=RANDOM_STATE
                        )
                        .values
                    )

                try:

                    _, p_norm = normaltest(
                        sample_values
                    )

                    if p_norm < 0.01:
                        normality_reasonable = False

                except Exception:
                    pass

            # ------------------------------------------------
            # ANOVA
            # ------------------------------------------------

            if normality_reasonable and (
                np.isnan(lev_p)
                or lev_p >= 0.05
            ):

                try:

                    stat_value, p_value = f_oneway(
                        *groups
                    )

                    eta_sq = eta_squared_from_groups(
                        groups
                    )

                    add_result(
                        "One-way ANOVA",
                        cat_col,
                        num_col,
                        stat_value,
                        p_value,
                        eta_sq,
                        "Parametric comparison of multiple groups"
                    )

                except Exception:
                    pass

            else:

                # ------------------------------------------------
                # KRUSKAL-WALLIS
                # ------------------------------------------------

                try:

                    stat_value, p_value = kruskal(
                        *groups
                    )

                    # Approximate epsilon-squared
                    n_total = sum(
                        len(g) for g in groups
                    )

                    k = len(groups)

                    epsilon_sq = (
                        stat_value - k + 1
                    ) / (
                        n_total - k
                    )

                    epsilon_sq = max(
                        0,
                        epsilon_sq
                    )

                    add_result(
                        "Kruskal-Wallis",
                        cat_col,
                        num_col,
                        stat_value,
                        p_value,
                        epsilon_sq,
                        "Non-parametric comparison of multiple groups"
                    )

                except Exception:
                    pass


# ============================================================
# 31. BARTLETT TEST
# ============================================================

"""
Bartlett's test is sensitive to non-normality.

Therefore we only apply it to selected groups when
normality appears reasonably acceptable.
"""

for cat_col in selected_group_categories:

    for num_col in selected_group_numerics:

        temp = df[
            [cat_col, num_col]
        ].dropna()

        groups = [
            group[num_col].values
            for _, group in temp.groupby(cat_col)
            if len(group) >= 10
        ]

        if len(groups) < 2:
            continue

        # Sample groups for speed
        sampled_groups = []

        for group in groups:

            if len(group) > MAX_TEST_SAMPLE:

                group = (
                    pd.Series(group)
                    .sample(
                        MAX_TEST_SAMPLE,
                        random_state=RANDOM_STATE
                    )
                    .values
                )

            sampled_groups.append(group)

        normality_ok = True

        for group in sampled_groups:

            if len(group) < 20:
                normality_ok = False
                break

            try:

                _, p_norm = normaltest(
                    group
                )

                if p_norm < 0.01:
                    normality_ok = False

            except Exception:

                normality_ok = False

        if normality_ok:

            try:

                stat_value, p_value = bartlett(
                    *sampled_groups
                )

                add_result(
                    "Bartlett variance test",
                    cat_col,
                    num_col,
                    stat_value,
                    p_value,
                    np.nan,
                    "Variance equality test under approximate normality"
                )

            except Exception:
                pass


# ============================================================
# 32. CATEGORICAL VS CATEGORICAL
# ============================================================

categorical_test_cols = [
    col
    for col in group_categorical_candidates
    if df[col].nunique(dropna=True) <= 10
]

# Limit pair combinations
categorical_test_cols = categorical_test_cols[:8]

for i, col1 in enumerate(categorical_test_cols):

    for col2 in categorical_test_cols[i + 1:]:

        temp = df[
            [col1, col2]
        ].dropna()

        if len(temp) < 30:
            continue

        contingency = pd.crosstab(
            temp[col1],
            temp[col2]
        )

        if (
            contingency.shape[0] < 2
            or contingency.shape[1] < 2
        ):
            continue

        # Avoid excessively sparse tables
        if (
            contingency.shape[0] > 10
            or contingency.shape[1] > 10
        ):
            continue

        try:

            chi2_value, p_value, dof, expected = (
                chi2_contingency(
                    contingency
                )
            )

            v = cramers_v(
                contingency
            )

            add_result(
                "Chi-square independence",
                col1,
                col2,
                chi2_value,
                p_value,
                v,
                "Association between categorical variables"
            )

        except Exception:
            pass


# ============================================================
# 33. STATISTICAL RESULTS TABLE
# ============================================================

statistical_tests = pd.DataFrame(
    statistical_results
)

if not statistical_tests.empty:

    statistical_tests[
        "adjusted_p_value"
    ] = benjamini_hochberg(
        statistical_tests["p_value"].values
    )

    statistical_tests[
        "significant_after_fdr"
    ] = (
        statistical_tests["adjusted_p_value"]
        < 0.05
    )

    statistical_tests[
        "significant_raw"
    ] = (
        statistical_tests["p_value"]
        < 0.05
    )

    statistical_tests[
        "effect_size_interpretation"
    ] = ""

    for idx, row in statistical_tests.iterrows():

        effect = row["effect_size"]

        if pd.isna(effect):

            interpretation = "Not applicable"

        elif row["test"] in [
            "Pearson correlation",
            "Spearman correlation"
        ]:

            abs_effect = abs(effect)

            if abs_effect < 0.10:
                interpretation = "Very weak"

            elif abs_effect < 0.30:
                interpretation = "Weak"

            elif abs_effect < 0.50:
                interpretation = "Moderate"

            else:
                interpretation = "Strong"

        elif "Cramér" in row["test"]:

            if effect < 0.10:
                interpretation = "Very small"

            elif effect < 0.30:
                interpretation = "Small"

            elif effect < 0.50:
                interpretation = "Moderate"

            else:
                interpretation = "Large"

        elif (
            "t-test" in row["test"]
            or "Mann-Whitney" in row["test"]
        ):

            if effect < 0.20:
                interpretation = "Negligible"

            elif effect < 0.50:
                interpretation = "Small"

            elif effect < 0.80:
                interpretation = "Medium"

            else:
                interpretation = "Large"

        else:

            if effect < 0.01:
                interpretation = "Very small"

            elif effect < 0.06:
                interpretation = "Small"

            elif effect < 0.14:
                interpretation = "Medium"

            else:
                interpretation = "Large"

        statistical_tests.loc[
            idx,
            "effect_size_interpretation"
        ] = interpretation

    statistical_tests = statistical_tests.sort_values(
        [
            "significant_after_fdr",
            "adjusted_p_value"
        ],
        ascending=[False, True]
    )

else:

    statistical_tests = pd.DataFrame(
        columns=[
            "test",
            "variable_1",
            "variable_2",
            "statistic",
            "p_value",
            "effect_size",
            "interpretation",
            "adjusted_p_value",
            "significant_after_fdr",
            "significant_raw",
            "effect_size_interpretation"
        ]
    )

statistical_tests.to_csv(
    TABLES_DIR / "statistical_tests.csv",
    index=False
)


# ============================================================
# 34. VIF ANALYSIS
# ============================================================

"""
VIF is calculated on a selected numerical feature set.

We avoid ID-like variables and constant variables.

For numerical stability, missing values are median-filled
only for VIF calculation. The original dataset is unchanged.
"""

vif_candidates = [
    col
    for col in analysis_numeric_cols
    if col not in id_like_cols
]

# Select highest-variance variables
vif_candidates = (
    numeric_df[vif_candidates]
    .var()
    .sort_values(ascending=False)
    .head(15)
    .index
    .tolist()
)

vif_df = numeric_df[
    vif_candidates
].copy()

vif_df = vif_df.replace(
    [np.inf, -np.inf],
    np.nan
)

vif_df = vif_df.fillna(
    vif_df.median(numeric_only=True)
)

# Remove remaining constant columns
vif_df = vif_df.loc[
    :,
    vif_df.nunique() > 1
]


def calculate_vif(dataframe):

    results = []

    columns = dataframe.columns.tolist()

    for col in columns:

        y = dataframe[col].values

        other_cols = [
            c
            for c in columns
            if c != col
        ]

        if len(other_cols) == 0:
            vif_value = 1.0

        else:

            X = dataframe[
                other_cols
            ].values

            try:

                model = LinearRegression()

                model.fit(
                    X,
                    y
                )

                r_squared = model.score(
                    X,
                    y
                )

                if r_squared >= 0.999999999:
                    vif_value = np.inf

                else:
                    vif_value = (
                        1 / (1 - r_squared)
                    )

            except Exception:

                vif_value = np.nan

        results.append({
            "variable": col,
            "VIF": vif_value
        })

    result_df = pd.DataFrame(
        results
    )

    def vif_status(value):

        if pd.isna(value):
            return "Unable to calculate"

        if value < 5:
            return "Generally acceptable"

        elif value <= 10:
            return "Possible concern"

        else:
            return "High multicollinearity"

    result_df["status"] = (
        result_df["VIF"]
        .apply(vif_status)
    )

    return result_df.sort_values(
        "VIF",
        ascending=False
    )


vif_analysis = calculate_vif(
    vif_df
)

vif_analysis.to_csv(
    TABLES_DIR / "vif_analysis.csv",
    index=False
)


# ============================================================
# 35. VIF AFTER TRANSFORMATION
# ============================================================

vif_after_df = transformed_numeric_df[
    vif_candidates
].copy()

vif_after_df = vif_after_df.replace(
    [np.inf, -np.inf],
    np.nan
)

vif_after_df = vif_after_df.fillna(
    vif_after_df.median(numeric_only=True)
)

vif_after_df = vif_after_df.loc[
    :,
    vif_after_df.nunique() > 1
]

vif_after_analysis = calculate_vif(
    vif_after_df
)

vif_before_after = vif_analysis[
    ["variable", "VIF"]
].rename(
    columns={"VIF": "VIF_before"}
).merge(
    vif_after_analysis[
        ["variable", "VIF"]
    ].rename(
        columns={"VIF": "VIF_after"}
    ),
    on="variable",
    how="outer"
)

vif_before_after.to_csv(
    TABLES_DIR / "vif_before_after.csv",
    index=False
)


# ============================================================
# 36. PLOT SETTINGS
# ============================================================

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})


def save_figure(fig, filename):

    fig.savefig(
        PLOTS_DIR / filename,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 37. FIGURE 1 - DATASET OVERVIEW
# ============================================================

fig = plt.figure(
    figsize=(15, 9)
)

gs = fig.add_gridspec(
    2,
    2,
    hspace=0.35,
    wspace=0.25
)

# Numerical vs categorical
ax1 = fig.add_subplot(gs[0, 0])

feature_counts = pd.Series({
    "Numerical": len(numeric_cols),
    "Categorical": len(categorical_cols),
    "Datetime": len(datetime_cols)
})

feature_counts.plot(
    kind="bar",
    ax=ax1
)

ax1.set_title(
    "Feature Type Distribution"
)

ax1.set_ylabel(
    "Number of Features"
)

ax1.set_xlabel("")

ax1.tick_params(
    axis="x",
    rotation=0
)


# Missing values
ax2 = fig.add_subplot(gs[0, 1])

missing_plot = missing_table[
    missing_table["missing_count"] > 0
].head(10)

if not missing_plot.empty:

    sns.barplot(
        data=missing_plot,
        x="missing_percentage",
        y="column",
        ax=ax2
    )

    ax2.set_title(
        "Top Missing-Value Percentages"
    )

    ax2.set_xlabel(
        "Missing (%)"
    )

    ax2.set_ylabel("")

else:

    ax2.text(
        0.5,
        0.5,
        "No missing values",
        ha="center",
        va="center",
        fontsize=14
    )

    ax2.set_axis_off()


# Dataset size
ax3 = fig.add_subplot(gs[1, 0])

size_data = pd.Series({
    "Rows": df.shape[0],
    "Columns": df.shape[1],
    "Duplicates": duplicate_count
})

size_data.plot(
    kind="bar",
    ax=ax3
)

ax3.set_title(
    "Dataset Structure"
)

ax3.set_ylabel(
    "Count"
)

ax3.set_xlabel("")

ax3.tick_params(
    axis="x",
    rotation=0
)


# Data quality
ax4 = fig.add_subplot(gs[1, 1])

quality_values = [
    len(constant_cols),
    len(id_like_cols),
    int(df.isna().sum().sum()),
    duplicate_count
]

quality_labels = [
    "Constant",
    "ID-like",
    "Missing Cells",
    "Duplicates"
]

ax4.bar(
    quality_labels,
    quality_values
)

ax4.set_title(
    "Data Quality Summary"
)

ax4.set_ylabel(
    "Count"
)

ax4.tick_params(
    axis="x",
    rotation=20
)

fig.suptitle(
    "OLIST Dataset Overview",
    fontsize=17,
    fontweight="bold"
)

save_figure(
    fig,
    "01_dataset_overview.png"
)


# ============================================================
# 38. FIGURE 2 - NUMERICAL DISTRIBUTIONS
# ============================================================

distribution_cols = (
    numeric_df.var()
    .sort_values(ascending=False)
    .head(MAX_DIST_VARS)
    .index
    .tolist()
)

n = len(distribution_cols)

if n > 0:

    rows = int(np.ceil(n / 2))

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(14, 4.2 * rows)
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(
        axes,
        distribution_cols
    ):

        series = numeric_df[col].dropna()

        # For very large variables, plotting a sample
        if len(series) > 20000:

            series = series.sample(
                20000,
                random_state=RANDOM_STATE
            )

        sns.histplot(
            series,
            kde=True,
            ax=ax,
            bins=30
        )

        ax.set_title(
            f"Distribution: {col}"
        )

        ax.set_xlabel(col)

        ax.set_ylabel(
            "Frequency"
        )

    for ax in axes[n:]:
        ax.remove()

    fig.suptitle(
        "Numerical Distribution Overview",
        fontsize=17,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        fig,
        "02_numerical_distributions.png"
    )


# ============================================================
# 39. FIGURE 3 - BOXPLOT OVERVIEW
# ============================================================

boxplot_cols = (
    numeric_df.var()
    .sort_values(ascending=False)
    .head(MAX_BOXPLOT_VARS)
    .index
    .tolist()
)

if boxplot_cols:

    box_data = numeric_df[
        boxplot_cols
    ].copy()

    # Standardize only for visual comparison.
    # Original values remain unchanged.
    box_scaled = (
        box_data
        - box_data.mean()
    ) / box_data.std()

    box_scaled = box_scaled.replace(
        [np.inf, -np.inf],
        np.nan
    )

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    sns.boxplot(
        data=box_scaled,
        orient="h",
        ax=ax
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Boxplot Overview - Standardized Scale",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Standardized Value"
    )

    ax.set_ylabel(
        "Numerical Variable"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "03_boxplot_overview.png"
    )


# ============================================================
# 40. FIGURE 4 - CATEGORICAL OVERVIEW
# ============================================================

categorical_plot_cols = []

for col in categorical_cols:

    if col in id_like_cols:
        continue

    unique_count = df[col].nunique(
        dropna=True
    )

    if 2 <= unique_count <= 15:

        categorical_plot_cols.append(col)

# Rank by number of observations represented
categorical_plot_cols = (
    categorical_plot_cols[:MAX_CAT_VARS]
)

if categorical_plot_cols:

    n = len(categorical_plot_cols)

    rows = int(np.ceil(n / 2))

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(15, 5 * rows)
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(
        axes,
        categorical_plot_cols
    ):

        counts = (
            df[col]
            .value_counts(dropna=False)
            .head(10)
        )

        labels = counts.index.astype(str)

        sns.barplot(
            x=counts.values,
            y=labels,
            ax=ax
        )

        ax.set_title(
            f"Top Categories: {col}"
        )

        ax.set_xlabel(
            "Number of Observations"
        )

        ax.set_ylabel(col)

    for ax in axes[n:]:
        ax.remove()

    fig.suptitle(
        "Categorical Feature Overview",
        fontsize=17,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        fig,
        "04_categorical_overview.png"
    )


# ============================================================
# 41. FIGURE 5 - CORRELATION HEATMAP
# ============================================================

if not pearson_matrix.empty:

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    sns.heatmap(
        pearson_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        ax=ax
    )

    ax.set_title(
        "Pearson Correlation Heatmap",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    fig.tight_layout()

    save_figure(
        fig,
        "05_correlation_heatmap.png"
    )


# ============================================================
# 42. FIGURE 6 - SKEWNESS ANALYSIS
# ============================================================

top_skew = skewness_analysis.head(
    MAX_SKEW_VARS
).sort_values(
    "skewness_before"
)

if not top_skew.empty:

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    sns.barplot(
        data=top_skew,
        x="skewness_before",
        y="variable",
        hue="classification",
        dodge=False,
        ax=ax
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.axvline(
        1,
        linestyle=":",
        linewidth=1
    )

    ax.axvline(
        -1,
        linestyle=":",
        linewidth=1
    )

    ax.set_title(
        "Highest Absolute Skewness",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Skewness"
    )

    ax.set_ylabel(
        "Variable"
    )

    ax.legend(
        title="Classification",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "06_skewness_analysis.png"
    )


# ============================================================
# 43. FIGURE 7 - KURTOSIS ANALYSIS
# ============================================================

top_kurtosis = kurtosis_analysis.head(
    MAX_KURTOSIS_VARS
).sort_values(
    "excess_kurtosis_before"
)

if not top_kurtosis.empty:

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    sns.barplot(
        data=top_kurtosis,
        x="excess_kurtosis_before",
        y="variable",
        hue="classification",
        dodge=False,
        ax=ax
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.axvline(
        3,
        linestyle=":",
        linewidth=1
    )

    ax.axvline(
        -3,
        linestyle=":",
        linewidth=1
    )

    ax.set_title(
        "Highest Absolute Excess Kurtosis",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Excess Kurtosis"
    )

    ax.set_ylabel(
        "Variable"
    )

    ax.legend(
        title="Classification",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "07_kurtosis_analysis.png"
    )


# ============================================================
# 44. FIGURE 8 - Q-Q PLOTS
# ============================================================

if normality_cols:

    n = len(normality_cols)

    rows = int(np.ceil(n / 2))

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(14, 4.8 * rows)
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(
        axes,
        normality_cols
    ):

        series = numeric_df[col].dropna()

        if len(series) > 5000:

            series = series.sample(
                5000,
                random_state=RANDOM_STATE
            )

        stats.probplot(
            series,
            dist="norm",
            plot=ax
        )

        ax.set_title(
            f"Q-Q Plot: {col}"
        )

        ax.set_xlabel(
            "Theoretical Quantiles"
        )

        ax.set_ylabel(
            "Ordered Values"
        )

    for ax in axes[n:]:
        ax.remove()

    fig.suptitle(
        "Normality Assessment - Q-Q Plots",
        fontsize=17,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        fig,
        "08_normality_qqplots.png"
    )


# ============================================================
# 45. FIGURE 9 - GROUP MEAN ANALYSIS
# ============================================================

group_plot_categories = (
    selected_group_categories[:4]
)

group_plot_numeric = (
    selected_group_numerics[:1]
)

if (
    group_plot_categories
    and group_plot_numeric
):

    target_num = group_plot_numeric[0]

    n = len(group_plot_categories)

    rows = int(np.ceil(n / 2))

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(15, 5 * rows)
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, cat_col in zip(
        axes,
        group_plot_categories
    ):

        plot_df = df[
            [cat_col, target_num]
        ].dropna()

        grouped = (
            plot_df
            .groupby(cat_col)[target_num]
            .agg(
                ["mean", "std", "count"]
            )
            .reset_index()
        )

        grouped = grouped.sort_values(
            "mean",
            ascending=False
        )

        sns.barplot(
            data=grouped,
            x="mean",
            y=cat_col,
            ax=ax
        )

        ax.set_title(
            f"Mean {target_num} by {cat_col}"
        )

        ax.set_xlabel(
            f"Mean {target_num}"
        )

        ax.set_ylabel(
            cat_col
        )

    for ax in axes[n:]:
        ax.remove()

    fig.suptitle(
        "Group-wise Mean / Dependence Analysis",
        fontsize=17,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        fig,
        "09_group_mean_analysis.png"
    )


# ============================================================
# 46. FIGURE 10 - STATISTICAL TEST SUMMARY
# ============================================================

if not statistical_tests.empty:

    significant_results = (
        statistical_tests[
            statistical_tests[
                "significant_after_fdr"
            ] == True
        ]
        .copy()
    )

    # If FDR leaves very few results,
    # display the strongest raw results.
    if len(significant_results) == 0:

        significant_results = (
            statistical_tests[
                statistical_tests[
                    "significant_raw"
                ] == True
            ]
            .copy()
        )

    significant_results = (
        significant_results.head(15)
    )

    if not significant_results.empty:

        plot_labels = (
            significant_results[
                "test"
            ]
            + " | "
            + significant_results[
                "variable_1"
            ].astype(str)
        )

        plot_data = pd.DataFrame({
            "label": plot_labels,
            "effect_size": significant_results[
                "effect_size"
            ].abs()
        })

        plot_data = plot_data.sort_values(
            "effect_size"
        )

        fig, ax = plt.subplots(
            figsize=(14, 9)
        )

        sns.barplot(
            data=plot_data,
            x="effect_size",
            y="label",
            ax=ax
        )

        ax.set_title(
            "Statistically Significant Results - Effect Sizes",
            fontsize=16,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Absolute Effect Size"
        )

        ax.set_ylabel(
            "Statistical Test / Variable"
        )

        fig.tight_layout()

        save_figure(
            fig,
            "10_statistical_tests.png"
        )


# ============================================================
# 47. BEFORE VS AFTER TRANSFORMATION SUMMARY
# ============================================================

if not transformation_table.empty:

    validation_table = transformation_table[
        [
            "variable",
            "skewness_before",
            "skewness_after",
            "absolute_skewness_before",
            "absolute_skewness_after"
        ]
    ].copy()

    validation_table.to_csv(
        TABLES_DIR / "before_after_validation.csv",
        index=False
    )


# ============================================================
# 48. REPORT TEXT FILE
# ============================================================

report_lines = []

report_lines.append(
    "OLIST DATASET - EDA AND STATISTICAL TESTING REPORT"
)

report_lines.append("=" * 60)

report_lines.append(
    f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns"
)

report_lines.append(
    f"Numerical features: {len(numeric_cols)}"
)

report_lines.append(
    f"Categorical features: {len(categorical_cols)}"
)

report_lines.append(
    f"Datetime features: {len(datetime_cols)}"
)

report_lines.append(
    f"Missing cells: {int(df.isna().sum().sum()):,}"
)

report_lines.append(
    f"Duplicate rows: {duplicate_count:,}"
)

report_lines.append("")

# Outliers
report_lines.append(
    "IMPORTANT OUTLIER FINDINGS"
)

if not outlier_summary.empty:

    top_outliers = (
        outlier_summary.head(5)
    )

    for _, row in top_outliers.iterrows():

        report_lines.append(
            f"- {row['variable']}: "
            f"{row['IQR_outlier_percentage']:.2f}% "
            f"IQR outliers; "
            f"{row['Z_score_outlier_percentage']:.2f}% "
            f"Z-score outliers"
        )

else:

    report_lines.append(
        "- No suitable numerical variables found."
    )

report_lines.append("")

# Skewness
report_lines.append(
    "HIGHLY SKEWED VARIABLES"
)

high_skew_report = skewness_analysis[
    skewness_analysis[
        "absolute_skewness"
    ] > 1
]

if not high_skew_report.empty:

    for col in high_skew_report[
        "variable"
    ].head(10):

        value = skewness_analysis.loc[
            skewness_analysis["variable"] == col,
            "skewness_before"
        ].iloc[0]

        report_lines.append(
            f"- {col}: skewness = {value:.3f}"
        )

else:

    report_lines.append(
        "- No highly skewed variables."
    )

report_lines.append("")

# Kurtosis
report_lines.append(
    "HIGH-KURTOSIS VARIABLES"
)

high_kurt_report = kurtosis_analysis[
    kurtosis_analysis[
        "excess_kurtosis_before"
    ] > 3
]

if not high_kurt_report.empty:

    for col in high_kurt_report[
        "variable"
    ].head(10):

        value = kurtosis_analysis.loc[
            kurtosis_analysis["variable"] == col,
            "excess_kurtosis_before"
        ].iloc[0]

        report_lines.append(
            f"- {col}: excess kurtosis = {value:.3f}"
        )

else:

    report_lines.append(
        "- No variables exceeded excess kurtosis of 3."
    )

report_lines.append("")

# Normality
report_lines.append(
    "NORMALITY FINDINGS"
)

if not normality_tests.empty:

    for _, row in normality_tests.iterrows():

        report_lines.append(
            f"- {row['variable']}: "
            f"{row['practical_normality']}; "
            f"Shapiro p={row['shapiro_p_value']:.4g}; "
            f"D'Agostino p={row['dagostino_p_value']:.4g}"
        )

else:

    report_lines.append(
        "- No variables available for normality analysis."
    )

report_lines.append("")

# Variance
report_lines.append(
    "IMPORTANT VARIANCE FINDINGS"
)

high_variance = (
    descriptive[
        ["variable", "variance"]
    ]
    .sort_values(
        "variance",
        ascending=False
    )
    .head(5)
)

for _, row in high_variance.iterrows():

    report_lines.append(
        f"- {row['variable']}: "
        f"variance = {row['variance']:.4g}"
    )

report_lines.append("")

# Group differences
report_lines.append(
    "IMPORTANT GROUP DIFFERENCES"
)

if not statistical_tests.empty:

    group_tests = statistical_tests[
        statistical_tests["test"].isin([
            "Independent t-test",
            "Mann-Whitney U",
            "One-way ANOVA",
            "Kruskal-Wallis"
        ])
        &
        (
            statistical_tests[
                "significant_after_fdr"
            ] == True
        )
    ]

    if not group_tests.empty:

        for _, row in group_tests.head(10).iterrows():

            report_lines.append(
                f"- {row['test']}: "
                f"{row['variable_1']} vs "
                f"{row['variable_2']} | "
                f"adjusted p="
                f"{row['adjusted_p_value']:.4g} | "
                f"effect="
                f"{row['effect_size']:.4g}"
            )

    else:

        report_lines.append(
            "- No group differences remained significant after FDR correction."
        )

else:

    report_lines.append(
        "- No statistical tests were available."
    )

report_lines.append("")

# Significant relationships
report_lines.append(
    "SIGNIFICANT STATISTICAL RELATIONSHIPS"
)

if not statistical_tests.empty:

    significant = statistical_tests[
        statistical_tests[
            "significant_after_fdr"
        ] == True
    ]

    report_lines.append(
        f"- {len(significant)} tests remained "
        f"significant after Benjamini-Hochberg FDR correction."
    )

else:

    report_lines.append(
        "- No statistical test results available."
    )

report_lines.append("")

# Multicollinearity
report_lines.append(
    "MULTICOLLINEARITY FINDINGS"
)

high_vif = vif_analysis[
    vif_analysis["VIF"] > 5
]

if not high_vif.empty:

    for _, row in high_vif.head(10).iterrows():

        report_lines.append(
            f"- {row['variable']}: "
            f"VIF = {row['VIF']:.3f} "
            f"({row['status']})"
        )

else:

    report_lines.append(
        "- No variables had VIF > 5 in the selected VIF set."
    )

report_lines.append("")

report_lines.append(
    f"Plots saved to: {PLOTS_DIR}"
)

report_lines.append(
    f"Tables saved to: {TABLES_DIR}"
)

report_lines.append(
    f"Reports saved to: {REPORTS_DIR}"
)

report_lines.append("")

report_lines.append(
    "IMPORTANT INTERPRETATION NOTE:"
)

report_lines.append(
    "Statistical significance does not automatically imply "
    "practical significance. Effect sizes, distributions, "
    "sample size, study design and domain context should also "
    "be considered."
)

report_text = "\n".join(
    report_lines
)

with open(
    REPORTS_DIR / "final_eda_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report_text
    )


# ============================================================
# 49. FINAL CONSOLE REPORT
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EDA REPORT")
print("=" * 60)

print(
    f"\nDataset shape: "
    f"{df.shape[0]:,} rows x {df.shape[1]} columns"
)

print(
    f"Numerical features: {len(numeric_cols)}"
)

print(
    f"Categorical features: {len(categorical_cols)}"
)

print(
    f"Datetime features: {len(datetime_cols)}"
)

print(
    f"Missing values: "
    f"{int(df.isna().sum().sum()):,}"
)

print(
    f"Duplicate rows: {duplicate_count:,}"
)


# Outlier findings
print("\nImportant outlier findings:")

if not outlier_summary.empty:

    for _, row in outlier_summary.head(5).iterrows():

        print(
            f"  - {row['variable']}: "
            f"{row['IQR_outlier_percentage']:.2f}% "
            f"IQR outliers"
        )

else:

    print("  - None")


# Skewness
print("\nHighly skewed variables:")

if not high_skew_report.empty:

    for col in high_skew_report[
        "variable"
    ].head(10):

        value = skewness_analysis.loc[
            skewness_analysis["variable"] == col,
            "skewness_before"
        ].iloc[0]

        print(
            f"  - {col}: {value:.3f}"
        )

else:

    print("  - None")


# Kurtosis
print("\nHigh-kurtosis variables:")

if not high_kurt_report.empty:

    for col in high_kurt_report[
        "variable"
    ].head(10):

        value = kurtosis_analysis.loc[
            kurtosis_analysis["variable"] == col,
            "excess_kurtosis_before"
        ].iloc[0]

        print(
            f"  - {col}: {value:.3f}"
        )

else:

    print("  - None")


# Normality
print("\nNormality findings:")

if not normality_tests.empty:

    for _, row in normality_tests.iterrows():

        print(
            f"  - {row['variable']}: "
            f"{row['practical_normality']}"
        )

else:

    print("  - None")


# Variance
print("\nImportant variance findings:")

for _, row in high_variance.iterrows():

    print(
        f"  - {row['variable']}: "
        f"variance={row['variance']:.4g}"
    )


# Group differences
print("\nImportant group differences:")

if not statistical_tests.empty:

    group_tests = statistical_tests[
        statistical_tests["test"].isin([
            "Independent t-test",
            "Mann-Whitney U",
            "One-way ANOVA",
            "Kruskal-Wallis"
        ])
        &
        (
            statistical_tests[
                "significant_after_fdr"
            ] == True
        )
    ]

    if not group_tests.empty:

        for _, row in group_tests.head(5).iterrows():

            print(
                f"  - {row['test']}: "
                f"{row['variable_1']} / "
                f"{row['variable_2']}"
            )

    else:

        print(
            "  - No group differences remained significant after FDR."
        )

else:

    print("  - None")


# Significant tests
print("\nSignificant statistical relationships:")

if not statistical_tests.empty:

    significant_count = int(
        statistical_tests[
            "significant_after_fdr"
        ].sum()
    )

    print(
        f"  - {significant_count} tests significant "
        f"after Benjamini-Hochberg FDR correction."
    )

else:

    print("  - None")


# VIF
print("\nMulticollinearity findings:")

if not high_vif.empty:

    for _, row in high_vif.head(5).iterrows():

        print(
            f"  - {row['variable']}: "
            f"VIF={row['VIF']:.3f}"
        )

else:

    print(
        "  - No VIF > 5 among selected variables."
    )


# Output paths
print("\nSaved plots:")
print(f"  {PLOTS_DIR}")

print("\nSaved tables:")
print(f"  {TABLES_DIR}")

print("\nSaved report:")
print(f"  {REPORTS_DIR / 'final_eda_report.txt'}")


# ============================================================
# 50. GITHUB COMMANDS
# ============================================================

print("\n")
print("=" * 60)
print("GITHUB COMMANDS")
print("=" * 60)

print("""
Run these commands from your VS Code terminal:

git status
git add .
git commit -m "Add complete Olist EDA analysis and visualizations"
git push origin main
""")


print("=" * 60)
print("EDA + STATISTICAL TESTING COMPLETED SUCCESSFULLY")
print("=" * 60)