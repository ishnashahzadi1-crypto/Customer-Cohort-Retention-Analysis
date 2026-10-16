# Customer Retention Report (v3)
 
## 1. Data summary
- Rows: **112,650** | Customers: **94,644** | Ever-repeat customers: **2,872 (3.03%)**
- Data ends: 2018-08-23 (found from order density)
- Average month-1 retention across big cohorts: **0.48%**
 
## 2. Target definition (fixed)
- Orders placed within the first **24h** are treated as ONE purchase (split checkouts).
- Label = another order in the following **90 days**; features are known at the end of the first session.
- With the old definition there were 1,547 positives; **37%** of them were same-session orders and are no longer counted -> 977 positives remain.
- Eligible customers: 75,724; positive rate 1.29%
- Dev 35,398 (488 positives, rate 1.38%) | Test 20,661 (262 positives, rate 1.27%)
 
## 3. Model selection (purged time-CV on Dev; out-of-fold metrics at the contact-budget threshold)
| model | cv_pr_auc_mean | cv_pr_auc_std | oof_pr_auc_pooled | oof_roc_auc | oof_accuracy | oof_balanced_accuracy | oof_precision | oof_recall | oof_f1 | oof_flagged_share |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.0201 | 0.0038 | 0.0175 | 0.504 | 0.9662 | 0.5134 | 0.0327 | 0.0471 | 0.0386 | 0.0207 |
| Logistic Regression | 0.02 | 0.0026 | 0.0192 | 0.5512 | 0.9525 | 0.5276 | 0.0364 | 0.0902 | 0.0519 | 0.0357 |
| Ensemble | 0.0194 | 0.0032 | 0.0185 | 0.554 | 0.9385 | 0.5244 | 0.0283 | 0.098 | 0.0439 | 0.0499 |
| Hist Gradient Boosting | 0.0193 | 0.0044 | 0.0182 | 0.539 | 0.9496 | 0.5223 | 0.0309 | 0.0824 | 0.045 | 0.0384 |
| Extra Trees | 0.0183 | 0.0036 | 0.0167 | 0.5379 | 0.9425 | 0.4975 | 0.0128 | 0.0392 | 0.0193 | 0.0442 |

Threshold = best F1 among thresholds that flag at most **5%** of customers. 10 random trials per model. The Ensemble is the average of the calibrated model scores.
 
**Selected model: Random Forest** (best CV PR-AUC). Accuracy is high mainly because ~98-99% of customers do not return; use PR-AUC, lift, precision and recall.
 
## 4. Test result (winner only, used once; 95% bootstrap CI)
- PR-AUC **0.0151** [0.0127, 0.0186] vs random baseline 0.0127
- ROC-AUC **0.5472** [0.5088, 0.5815]
- Accuracy 0.7017 | Balanced accuracy 0.5306 | Precision 0.0153 | Recall 0.3550 | F1 0.0293 | flags 29.5% of customers
- Top 10% scored customers: precision 0.018, captures 14.1% of repeaters, lift **1.41x** [1.04, 1.90]
- Brier (calibrated) 0.01252 vs base-rate predictor 0.01252
- Top features (Random Forest, held-out fold): freight_sum, distance_km, name_len_mean, promised_days, category
 
## 4b. Plots
- `figures/eda/` (17 files): cohorts, orders trend, orders per customer, days-to-2nd-order, repeat rate by value / freight / distance / promised days / category / state / payment / installments / month, correlation, distributions, missing values.
- `figures/ml/` (13 files): CV comparison, metrics bars + table, PR/ROC curves, confusion matrix, gains, lift, calibration, score distribution, threshold analysis, importance, fold stability.
 
## 5. Honest limitations
- Repeat purchasing is rare, so precision/F1 stay low by construction; a model cannot create signal that the first-order data does not contain.
- **Dataset is small** (112,650 rows): few positives, every metric is noisy.
- Test positives are few: read the confidence intervals, not point estimates.
- Dev and Test base rates can differ (see eda/10); the threshold may need re-tuning on fresh data.
- Final model is refit on all Dev, while calibration/threshold come from smaller fold models: a small mismatch.
- 'No return yet' is not permanent churn.
