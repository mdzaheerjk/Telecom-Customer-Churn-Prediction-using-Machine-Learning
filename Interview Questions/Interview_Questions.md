# 📡 Telecom Customer Churn Prediction — Top 50 Interview Questions & Answers

[![Live Demo](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://telecom-customer-churn-prediction-using-machine-learning.streamlit.app/)
![Questions](https://img.shields.io/badge/Questions-50-blue?style=flat-square)
![Level](https://img.shields.io/badge/Level-Beginner_→_Senior-8A2BE2?style=flat-square)
![Verified](https://img.shields.io/badge/Verified_against_repo-28_Oct_2026-2EA043?style=flat-square)

> **Interview guide for `mdzaheerjk/Telecom-Customer-Churn-Prediction-using-Machine-Learning`.**
> Every answer is tied to this repo's actual code, notebooks and data — the way a real interviewer probes a portfolio project: *what did you do, why, what are its weaknesses, and what would you change?*
> Figures were re-checked against `Data/Customer-Churn.csv`, `Notebooks/MODEL.ipynb`, `app.py` and the deployed `Ada_boost_churn_model.pkl`. Figures labelled **re-run** come from stratified 80/20 splits over 10 seeds with scikit-learn 1.5.2 (the pickle's training version); the notebook itself used a single unseeded split.

---

## 🎤 The 30-Second Pitch

> *"I built an end-to-end churn system on the 7,043-customer Telco dataset. After cleaning, I engineered 34 features and benchmarked decision trees, random forests, XGBoost, LightGBM, CatBoost and AdaBoost against several imbalance strategies — SMOTE, ADASYN, SMOTEENN and cost-sensitive weighting. I deployed a cost-weighted AdaBoost that catches about 80% of churners at roughly 51–53% precision, wrapped it in a non-root Docker container and a Streamlit app that returns a risk score, a risk tier and rule-based retention actions. The main thing I learned is that all the models land at about 0.84 ROC-AUC, so the real engineering is in calibration, thresholds tied to retention cost, and keeping training and serving consistent."*

---

## 🔢 Numbers to Know Cold

| Fact | Value |
| :--- | :--- |
| Dataset | 7,043 rows × 21 columns (19 features + `customerID` + `Churn`) |
| Churners / churn rate | 1,869 / **26.54%** — majority-class accuracy is **73.4%** |
| Dirty `TotalCharges` | 11 blank strings, all with `tenure = 0` → 7,032 clean rows |
| Final feature count | **34** (3 numeric + 31 dummies, `tenure` replaced by 6 cohort bins) |
| Split used in notebook | 80/20, **no `random_state`, no `stratify`** → 5,625 train (4,160 / 1,465) and 1,407 test |
| Class weight | `4160 / 1465 =` **2.84** |
| Deployed model | `AdaBoostClassifier`: 50 depth-1 stumps, `learning_rate=1.0`, `SAMME.R`, trained with `sample_weight` |
| Notebook test result (single split) | Accuracy 0.738 · Precision 0.53 · **Recall 0.82** · F1 0.64 · confusion matrix `[[706, 297], [71, 333]]` |
| **Re-run, 10 stratified seeds** | ROC-AUC **0.844 ± 0.008** · PR-AUC 0.656 · Recall 0.805 ± 0.015 · Precision 0.514 · F1 0.627 · Accuracy 0.746 · Brier 0.244 |
| Brier of "always predict 26.5%" | 0.195 — the deployed model's raw scores (0.244) are *worse* than this, i.e. they are not probabilities |
| Tuning | RandomizedSearchCV 50×5-fold (best CV F1 0.630) · Optuna 100 trials (best CV F1 0.6335) |

---

## 🔍 Repo Audit — What an Interviewer Will Find (and What to Say)

Strong candidates raise these themselves. Each row is a real finding from reading the code and re-running it.

| # | Finding | Where | What to say |
| :-: | :--- | :--- | :--- |
| 1 | Split has no `random_state` / `stratify` | `MODEL.ipynb` cells 9, 31 | *"Results vary ±0.01–0.02 per run, so I treat 0.82 recall as ≈0.80 ± 0.02 and report repeated stratified CV."* |
| 2 | Model chosen on the same test set used to compare ~15 models | `MODEL.ipynb` | *"Winner's-curse bias; I'd carve out a validation set or use nested CV and touch the test set once."* |
| 3 | App sidebar shows **accuracy 81.2%**; the notebook never computes ROC-AUC | `app.py` | *"AUC 0.84 reproduces; 81.2% accuracy does not — the weighted model scores ≈74–75%. I'd compute metrics in a script and load them from a JSON artifact."* |
| 4 | "Churn probability" is a score squeezed into ≈0.47–0.52 | `app.py` tiers | *"SAMME.R averages stump votes, so probabilities collapse towards 0.5. Ranking is excellent, calibration is poor — I'd calibrate before showing percentages."* |
| 5 | README tiers (35% / 60%) differ from code tiers (45% / 50%) | README vs `app.py` | *"README thresholds were never reachable with this score range; code thresholds were tuned to it. I'd derive tiers from calibrated probabilities and expected cost."* |
| 6 | Scaler isn't saved; app refits it on all 7,032 rows | `build_preprocessor` | *"Tiny training-serving skew (≈0.6% label flips in a re-run). Fix: persist one sklearn `Pipeline`."* |
| 7 | "Top Risk Feature Breakdown" is a hard-coded global chart | `app.py` | *"It's the model's global stump-count importance, not a per-customer explanation. SHAP would give local attribution."* |
| 8 | UI claims like "over 52% of churn" and "< 12 ms" are hard-coded | `app.py` | *"Recomputed: 1–12-month customers without dependents churn at 51.0% and make up 47.6% of all churners. Latency should be measured, not asserted."* |
| 9 | `MinMaxScaler` fitted on already-standardised data | `MODEL.ipynb` cell 37 | *"Harmless for trees but sloppy; scaling belongs in a Pipeline fit on train only."* |
| 10 | README benchmark rows differ from notebook outputs (e.g. Random Forest) | README vs notebook | *"Numbers came from separate unseeded runs; I now regenerate the table from a script with fixed seeds."* |
| 11 | README/docs describe env-var config, logging and a multi-stage Dockerfile; `app.py` reads no env vars and the Dockerfile is single-stage | docs vs code | *"Docs got ahead of the code — I aligned them and listed config/logging as a patch."* |
| 12 | Optuna's final refit drops `random_state` / `eval_metric` | `MODEL.ipynb` cell 62 | *"`best_params` only contains tuned keys; fixed params must be re-added for reproducibility."* |

---

## 📑 Progress Tracker

| Part | Focus | Questions | Done |
| :--- | :--- | :---: | :---: |
| **1** | Project story, data & EDA | Q1 – Q10 | `[ ]` |
| **2** | Preprocessing, features & class imbalance | Q11 – Q22 | `[ ]` |
| **3** | Models, math & tuning | Q23 – Q34 | `[ ]` |
| **4** | Evaluation, calibration, explainability & business | Q35 – Q42 | `[ ]` |
| **5** | Production, MLOps & system design | Q43 – Q50 | `[ ]` |

---

# Part 1 — Project Story, Data & EDA (Q1–Q10)

### 🟢 Q1. Walk me through this project in two minutes.

**Topic:** Problem framing · **Interviewer wants:** structure, ownership, honesty about limits.

<details>
<summary><b>💡 Answer & tips</b></summary>

**Structure (Problem → Data → Model → Deployment → Lesson):**
1. *Problem:* telecom churn is costly; the goal is to rank customers by risk so retention offers go to the right people.
2. *Data:* 7,043 customers, 26.5% churn, mostly categorical service/billing fields.
3. *Model:* benchmarked 6+ algorithms × 5 imbalance strategies; deployed cost-weighted AdaBoost (recall ≈0.80, precision ≈0.51).
4. *Deployment:* Streamlit app, hardened Docker image, deployment guides for Streamlit Cloud, Docker and AWS.
5. *Lesson:* ROC-AUC plateaus at ≈0.84 for every model — the leverage is calibration, thresholds and consistency between training and serving.

**Tip:** end by naming one weakness and its fix (e.g. "scores aren't calibrated; next step is isotonic calibration"). It signals maturity and steers the follow-up questions to ground you control.
</details>

---

### 🟢 Q2. What business decision does this model drive, and how do you know it's worth building?

**Topic:** Business value · **Interviewer wants:** you connect a metric to money.

<details>
<summary><b>💡 Answer & tips</b></summary>

The model decides **who gets a retention contact**. It pays off when

$$\text{expected gain per contacted customer} = p \cdot s \cdot V - c > 0$$

with $p$ = true churn probability, $s$ = chance the offer saves the customer, $V$ = value retained, $c$ = offer cost.

*Illustration (assumptions, not data):* with $V = \$1{,}200$ (the app's acquisition-cost figure), $s = 30\%$, $c = \$50$, contact anyone with $p > 50/360 \approx 14\%$. That is far below a 0.5 cut-off — which is why the cut-off must come from costs, not from a default.

**Trap:** "99% accuracy"-style answers. Say you evaluate in **dollars saved per 1,000 customers contacted** and via a holdout campaign test.
</details>

---

### 🟢 Q3. Describe the dataset and the target. Is it imbalanced?

**Topic:** Data understanding

<details>
<summary><b>💡 Answer & tips</b></summary>

- 7,043 customers, 21 columns: demographics, services (phone, internet, add-ons, streaming), account (contract, billing, payment), `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn`.
- Churn = 26.54% (1,869). A dummy model that always says "No" scores **73.4% accuracy**.
- Imbalance ratio ≈ **2.8 : 1** — *moderate*, not extreme. That matters: resampling gains are modest, which the benchmark confirms (every strategy lands at minority F1 0.61–0.64).

**Follow-up:** *"Would you call that highly imbalanced?"* → No; at 2.8:1 class weights or threshold tuning are usually enough, and SMOTE-style synthesis is rarely worth the risk.
</details>

---

### 🟢 Q4. `TotalCharges` loaded as a string. Why, and how did you fix it? What happens at `tenure = 0` in the app?

**Topic:** Data cleaning

<details>
<summary><b>💡 Answer & tips</b></summary>

**Cause:** 11 rows contain a blank string `' '`, so pandas infers `object`. All 11 have `tenure = 0` — brand-new customers not yet billed.

```python
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
telco.dropna(how='any', inplace=True)   # 11 rows (0.16%) → 7,032 remain
```

**Why drop vs impute?** They are structurally missing (no bill yet), a tiny fraction, and imputing 0 would add a pattern the model never sees at training time.

**Live-app edge case:** the slider allows `tenure = 0`, but training never saw it. The app defaults `TotalCharges` to `MonthlyCharges × max(tenure, 1)` and `pd.cut(..., include_lowest=True)` places it in the `1-12` bin. Better: validate inputs (`tenure ≥ 1`) or document that scores for new sign-ups are out-of-distribution.
</details>

---

### 🟡 Q5. Which features could leak the target, and why drop `customerID`?

**Topic:** Leakage · **Interviewer wants:** you think about *when* each feature is measured.

<details>
<summary><b>💡 Answer & tips</b></summary>

- `customerID` is a unique key (7,043 distinct values) — pure memorisation fuel, no signal.
- **Subtle leakage:** in this snapshot, `tenure` and `TotalCharges` for churners are measured **at exit**, so the data partly encodes "how long until they left". In production you score *active* customers whose tenure is still growing.
- The clean design is a **point-in-time dataset**: features as of date *t*, label = "churned in (t, t + 30 days]".

**Say:** *"The public dataset has no timestamps, so I can't do temporal validation; with real data I'd build snapshots and split by time."*
</details>

---

### 🟢 Q6. What were your key EDA findings? (Give numbers.)

**Topic:** EDA · **Interviewer wants:** insight, not chart descriptions.

<details>
<summary><b>💡 Answer & tips</b></summary>

| Segment | Churn rate |
| :--- | :---: |
| Month-to-month / 1-year / 2-year contract | **42.7%** / 11.3% / 2.8% |
| Tenure 1–12 months → 61–72 months | **47.7%** → 6.6% |
| Fiber optic / DSL / no internet | **41.9%** / 19.0% / 7.4% |
| Electronic check vs credit-card autopay | **45.3%** vs 15.2% |
| No tech support vs tech support | 41.6% vs 15.2% |
| Senior vs non-senior | 41.7% vs 23.6% |
| No dependents vs dependents | 31.3% vs 15.5% |

**Headline:** 55% of all churners leave in the first 12 months. Average monthly charge is $74.4 for churners vs $61.3 for stayers.

**So what?** Focus on first-year onboarding and nudging month-to-month customers to term contracts and autopay.
</details>

---

### 🟡 Q7. Fiber-optic customers churn more. Is fiber causing churn?

**Topic:** Confounding & interactions

<details>
<summary><b>💡 Answer & tips</b></summary>

Not necessarily. Fiber is **priced higher** and is sold more often to month-to-month, electronic-check customers — all churn drivers.

**Interaction evidence:** fiber customers *without* tech support churn at **49.4%**; fiber customers *with* it churn at **22.6%**. The risk sits in the combination, so the retention action is "bundle support with fiber", not "stop selling fiber".

**Rigour:** a stratified table or a model with both terms shows association; only a randomised offer shows causation (see Q41).
</details>

---

### 🟡 Q8. Did you check multicollinearity? Does it matter for your model?

**Topic:** Statistics

<details>
<summary><b>💡 Answer & tips</b></summary>

Yes: `TotalCharges` correlates **0.83** with `tenure` and 0.65 with `MonthlyCharges` (it is roughly their product).

- **Linear/logistic models:** coefficients become unstable → check VIF, drop or combine features, or use L2.
- **Tree ensembles:** unaffected for prediction; each split picks one feature. But **importance gets split** between correlated features, which affects interpretation.
- Note: you replaced `tenure` with `tenure_bin` but kept `TotalCharges`, which still proxies tenure — the information was reduced, not removed.
</details>

---

### 🟡 Q9. Voluntary vs involuntary churn — and what is the prediction horizon here?

**Topic:** Problem definition

<details>
<summary><b>💡 Answer & tips</b></summary>

- **Voluntary:** customer chooses to leave (price, competitor, service). **Involuntary:** service ends for non-payment/failed cards.
- The dataset says "customers who left last month" — mostly voluntary, but the label doesn't distinguish.
- **Horizon is undefined** in the public data. In production: define it (e.g. 30 days) and consider **survival analysis** (Cox / discrete-time hazard) when *when* matters, not just *whether*.
</details>

---

### 🟢 Q10. What data-quality checks did you run besides `TotalCharges`?

**Topic:** Data hygiene

<details>
<summary><b>💡 Answer & tips</b></summary>

- Null/duplicate/ID-uniqueness checks; value-set checks for every categorical column.
- **Inconsistent encodings:** `SeniorCitizen` is 0/1 while other flags are `Yes/No`; add-on columns use a third level `"No internet service"` that duplicates information in `InternetService = No`.
- Range checks: `tenure` 0–72, `MonthlyCharges` ≈ 18–119.
- In production add a **schema contract** (Pydantic / Pandera) so bad inputs fail loudly instead of becoming silent zeros after `reindex(fill_value=0)`.
</details>

---

# Part 2 — Preprocessing, Features & Class Imbalance (Q11–Q22)

### 🟢 Q11. Why bin `tenure`, and what are the bins?

**Topic:** Feature engineering

<details>
<summary><b>💡 Answer & tips</b></summary>

Bins `[0,12,24,36,48,60,72]` → labels `1-12 … 61-72` with `include_lowest=True`. Counts: 2,175 / 1,024 / 832 / 762 / 832 / 1,407. Observed churn falls monotonically from 47.7% to 6.6%.

**Honest trade-off:** trees can find their own cut-points, so binning mostly helps interpretability and linear models. It *loses* resolution (month 12 and month 13 are treated as different cohorts, month 1 and month 12 as the same). Validate with CV: keep raw `tenure` vs bins and compare PR-AUC.
</details>

---

### 🟢 Q12. What is the dummy-variable trap? How did you arrive at 34 features?

**Topic:** Encoding

<details>
<summary><b>💡 Answer & tips</b></summary>

With one-hot encoding, a full set of dummies sums to 1 → perfectly collinear with the intercept. `drop_first=True` removes one level per column.

**34 = 3 numeric** (`SeniorCitizen`, `MonthlyCharges`, `TotalCharges`) **+ 31 dummies:** four binary flags (gender, Partner, Dependents, PhoneService) + 8 three-level columns × 2 + Contract 2 + PaperlessBilling 1 + PaymentMethod 3 + tenure_bin 5.

**Caveat:** trees don't need `drop_first`; it matters for linear models and is harmless otherwise.
</details>

---

### 🟡 Q13. Does AdaBoost need feature scaling? Why is scaling in your pipeline anyway?

**Topic:** Scaling

<details>
<summary><b>💡 Answer & tips</b></summary>

**No.** A stump splits on `x_j ≤ threshold`; any monotonic rescaling just moves the threshold. Scaling matters for linear models, SVMs, KNN, neural nets and penalised models.

**Real issues in the repo:**
1. Notebook applies `StandardScaler` to all models, then fits a `MinMaxScaler` *on the already-standardised data* (cell 37).
2. The training scaler is **never saved**. The app refits one on all 7,032 rows, but the stumps' thresholds live in the *training split's* scaled units.

A re-run shows the effect is small (max score shift ≈0.005, about 0.6% of labels flip near 0.5) — but it is avoidable: persist a single `Pipeline`, or drop the scaler for tree models.
</details>

---

### 🟢 Q14. Why does accuracy fail here? Show it with your own results.

**Topic:** Metrics

<details>
<summary><b>💡 Answer & tips</b></summary>

- Baseline "always No" = **73.4%** accuracy.
- Plain decision tree: accuracy 0.76 but churn recall 0.51, F1 0.55.
- Deployed weighted AdaBoost: accuracy **0.738** (lower!) but recall **0.82**.

The "worse" accuracy is the better business model: it deliberately trades false alarms for catching churners. Report precision/recall/F1 for the minority class, PR-AUC, and a cost-based metric.
</details>

---

### 🟡 Q15. Compare SMOTE, ADASYN and SMOTEENN. What did your experiments show?

**Topic:** Resampling

<details>
<summary><b>💡 Answer & tips</b></summary>

- **SMOTE:** interpolates between a minority point and one of its *k* neighbours. **ADASYN:** generates more samples near hard (majority-dense) regions. **SMOTEENN:** SMOTE then Edited Nearest Neighbours removes noisy points from both classes.
- **Your results (single split, minority class):** XGB+SMOTE F1 0.61 · XGB+ADASYN 0.62 · DT+SMOTEENN 0.61 · RF+SMOTEENN 0.63 · class-weighted models 0.62–0.64.
- **Interpretation:** they all slide along the *same* precision-recall curve; resampling doesn't create information.
- **Risk with this data:** features are one-hot dummies. SMOTE interpolates them into values like 0.37, which are not real customers; use `SMOTENC` or avoid synthesis.
</details>

---

### 🟡 Q16. What is cost-sensitive learning, and why did weighting beat resampling for deployment?

**Topic:** Imbalance strategy

<details>
<summary><b>💡 Answer & tips</b></summary>

You pass `sample_weight = 2.84` for churners (`4160/1465`), so each missed churner costs ~2.8× a false alarm in the training loss. Benefits: no synthetic rows, no resampling leakage risk, one line of code, and a model that stays deployable with a plain scikit-learn dependency.

**Cost you must mention:** weighting shifts the *effective class prior* to ~50%, so raw output scores are **not calibrated to the real 26.5% churn rate** (see Q38). Resampling does the same.
</details>

---

### 🔴 Q17. Where in the pipeline must resampling happen, and how would you do it inside cross-validation?

**Topic:** Leakage

<details>
<summary><b>💡 Answer & tips</b></summary>

Resample **only the training portion**, *after* the split. Your notebook does this (SMOTEENN on `x_train`, test untouched) ✅.

For CV, resampling must happen **inside each fold**:

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
pipe = Pipeline([("smote", SMOTE(random_state=42)), ("clf", XGBClassifier())])
cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="average_precision")
```

Resampling before CV puts near-copies of validation rows in training and inflates scores.
</details>

---

### 🟡 Q18. Your `train_test_split` has no `random_state` and no `stratify`. Why does it matter?

**Topic:** Evaluation rigour

<details>
<summary><b>💡 Answer & tips</b></summary>

- **Not reproducible:** every run gives different numbers (README and notebook already disagree on several rows).
- **Not stratified:** the test set's churn share drifts (396/1,409 vs 404/1,407 in two runs).
- **Noise size:** with ≈374 positives in a 20% test set, recall has a standard error of $\sqrt{0.8 \cdot 0.2 / 374} \approx 0.02$ — a ±0.04 95% interval. Differences like 0.79 vs 0.82 are **not significant**. Over 10 seeds the deployed recipe gives recall 0.805 ± 0.015.

**Fix:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` for the final hold-out, and `RepeatedStratifiedKFold(5, 3)` for model comparison.
</details>

---

### 🔴 Q19. What is training-serving skew, and where is it in this repo?

**Topic:** Production consistency

<details>
<summary><b>💡 Answer & tips</b></summary>

Training and inference must apply **identical, fitted** transformations.

| Risk | Status in repo |
| :--- | :--- |
| Scaler fit on all data in the app vs train split in the notebook | ⚠️ present (small effect, see Q13) |
| Preprocessing re-implemented by hand in `app.py` | ⚠️ duplicated logic can drift |
| Column order / missing dummies | ✅ handled by `reindex(columns=template, fill_value=0)` |
| `tenure = 0` unseen in training | ⚠️ out-of-distribution input |

**Fix:** one `Pipeline(ColumnTransformer(...), model)` saved with `joblib`, plus a **parity test** that scores 100 held-out rows through both notebook and app code paths and asserts equal output.
</details>

---

### 🟡 Q20. How does the app turn one customer into the 34-column vector?

**Topic:** Inference pipeline

<details>
<summary><b>💡 Answer & tips</b></summary>

1. Build a one-row DataFrame; bin `tenure` with the same bins; drop `tenure`.
2. `pd.get_dummies(df_in)` **without** `drop_first` — with one row, `drop_first` would delete the only observed level.
3. `reindex(columns=template_columns, fill_value=0)` — absent dummy levels become 0, which equals the dropped baseline level.
4. `scaler.transform(...)` → `model.predict_proba(...)`.

**Weakness:** this re-implements encoding by hand. `OneHotEncoder(drop="first", handle_unknown="ignore")` inside a `ColumnTransformer` gives the same result, serialised with the model.
</details>

---

### 🟡 Q21. What encoding alternatives did you consider?

**Topic:** Encoding

<details>
<summary><b>💡 Answer & tips</b></summary>

- **One-hot** (used): safe for low-cardinality features; all features here have ≤ 4 levels.
- **Ordinal** for `Contract` (month-to-month < 1y < 2y) — compact and monotone.
- **Target encoding:** powerful for high-cardinality columns but leaks unless computed out-of-fold; unnecessary here.
- **Native categoricals** (CatBoost/LightGBM): you passed one-hot columns to CatBoost, which bypasses its main advantage (ordered target statistics).
- Collapse `"No internet service"` into `"No"` plus an `has_internet` flag to remove redundancy.
</details>

---

### 🔴 Q22. What features would you add, and how would you prove they help?

**Topic:** Feature engineering

<details>
<summary><b>💡 Answer & tips</b></summary>

- `avg_monthly_spend = TotalCharges / max(tenure, 1)` and `price_change = MonthlyCharges − avg_monthly_spend` (recent price hikes).
- `n_addons` (count of security/backup/protection/support) and `has_support`.
- Explicit interactions: `fiber & no_support`, `month_to_month & e_check`.
- With real data: usage trends, support-call counts, complaint tickets, bill shock, contract end-date proximity.

**Proof:** add one feature at a time, compare **repeated-CV PR-AUC** (not in-sample importance), and keep it only if the gain exceeds the CV standard deviation (≈0.01).
</details>

---
# Part 3 — Models, Math & Tuning (Q23–Q34)

### 🟡 Q23. Show your benchmark. Why did AdaBoost win — and is that win real?

**Topic:** Model selection · **Interviewer wants:** you can read your own results critically.

<details>
<summary><b>💡 Answer & tips</b></summary>

**Notebook results, minority class, one unseeded 80/20 split (n = 1,407):**

| Model / imbalance strategy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Decision Tree (baseline) | 0.59 | 0.51 | 0.55 |
| XGBoost (unweighted) | 0.61 | 0.53 | 0.57 |
| XGBoost + SMOTE / + ADASYN | 0.62 / 0.63 | 0.61 / 0.61 | 0.61 / 0.62 |
| XGBoost + `scale_pos_weight` | 0.57 | 0.68 | 0.62 |
| Random Forest (balanced, depth 6) | 0.51 | 0.80 | 0.62 |
| CatBoost (balanced) | 0.54 | 0.75 | 0.63 |
| LightGBM (`scale_pos_weight`) | 0.55 | 0.77 | 0.64 |
| XGBoost (RandomizedSearchCV) | 0.53 | 0.81 | 0.64 |
| **AdaBoost (cost-weighted)** | **0.53** | **0.82** | **0.64** |

**Honest reading:** every weighted/tuned model sits at F1 0.62–0.64, inside the ±0.02 noise of a 1,407-row test split (Q18). AdaBoost did not *beat* the others — it **tied** them while being the simplest to ship: one scikit-learn dependency, a 33 KB pickle, no C++ boosting runtime.

**Better framing:** *"I chose it for operational simplicity at equal accuracy. Its weaknesses are poor probability calibration and additive-only structure; if calibrated probabilities mattered more, I'd pick a log-loss model."*
</details>

---

### 🔴 Q24. Explain AdaBoost mathematically. What exactly did you deploy?

**Topic:** Algorithms

<details>
<summary><b>💡 Answer & tips</b></summary>

Start with equal sample weights $w_i$. For round $t$:

1. Fit a weak learner $h_t$ on the weighted data; compute weighted error $\varepsilon_t = \sum_i w_i\,\mathbb{1}[h_t(x_i) \ne y_i]$.
2. Learner weight: $\alpha_t = \tfrac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}$ (scikit-learn's discrete SAMME drops the ½).
3. Re-weight: $w_i \leftarrow w_i \exp(\alpha_t \mathbb{1}[h_t(x_i)\ne y_i])$ and normalise — misclassified points matter more next round.
4. Predict $H(x) = \operatorname{sign}\sum_t \alpha_t h_t(x)$.

This is forward stagewise minimisation of the **exponential loss** $e^{-yF(x)}$.

**What you deployed (read from the pickle):** 50 depth-1 stumps, `learning_rate=1.0`, `algorithm="SAMME.R"` (the real-valued variant that uses each stump's class probabilities), trained with `sample_weight` = 2.84 on churners. Only **17 of 34** features are ever used; `feature_importances_` moves in steps of 0.02 (= 1/50) because it is essentially a stump count — **18 stumps split on `TotalCharges`, 13 on `MonthlyCharges`**.

**Why scores look like 0.47–0.52:** scikit-learn averages the stumps' contributions instead of summing them, so the final score concentrates near 0.5 (Q38). SAMME.R was deprecated in 1.4 and removed in 1.6, so a retrain on a newer version uses SAMME with a different score scale — **recalibrate and re-derive thresholds after any upgrade.**
</details>

---

### 🟡 Q25. AdaBoost vs Gradient Boosting vs XGBoost — what's the difference?

**Topic:** Boosting family

<details>
<summary><b>💡 Answer & tips</b></summary>

| | AdaBoost | Gradient Boosting | XGBoost |
| :--- | :--- | :--- | :--- |
| Mechanism | Re-weights samples | Fits the negative gradient of any differentiable loss | Same, with 2nd-order (Newton) steps |
| Loss | Exponential | Log-loss, squared error, … | Log-loss etc. + L1/L2 + γ penalties |
| Noise sensitivity | **High** (exponential loss punishes mislabelled points) | Moderate | Lower (shrinkage, subsampling, regularisation) |
| Probabilities | Poorly calibrated | Reasonable | Reasonable |

AdaBoost is gradient boosting with exponential loss (Friedman). Churn labels are noisy (people leave for reasons not in the data), which is one reason AdaBoost needed shallow stumps here.
</details>

---

### 🔴 Q26. Stumps are depth-1. Can your model learn interactions like "fiber + no tech support"?

**Topic:** Model capacity · **Interviewer wants:** a precise answer, not "it models complex boundaries".

<details>
<summary><b>💡 Answer & tips</b></summary>

**No — not directly.** A sum of single-feature stumps is an **additive model**, $F(x)=\sum_j f_j(x_j)$ (a GAM-like structure). It only captures the fiber × no-support effect (49.4% vs 22.6% churn) *additively*, through the correlated one-hot columns.

**Check it, don't assume it** (repeated 5×3 stratified CV on the same 34 features):

| Model | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |
| AdaBoost, depth-1 stumps ×50 | 0.844 | 0.659 |
| AdaBoost, **depth-2** trees ×50 | 0.822 | 0.613 |
| Gradient Boosting, depth 3 ×200 | 0.846 | 0.663 |

**Conclusion:** interactions add essentially nothing here (deeper AdaBoost overfits; gradient boosting gains +0.002). The dataset is close to additive — which is also why logistic regression keeps up (Q29). Optuna independently chose `max_depth = 3`, the low end of its range.
</details>

---

### 🔴 Q27. How does XGBoost use a second-order Taylor expansion?

**Topic:** Algorithms

<details>
<summary><b>💡 Answer & tips</b></summary>

For round $t$ it approximates the loss around the current prediction:

$$\mathcal{L}^{(t)} \approx \sum_i \Big[g_i f_t(x_i) + \tfrac12 h_i f_t(x_i)^2\Big] + \Omega(f_t),\quad g_i=\partial_{\hat y}\ell,\; h_i=\partial^2_{\hat y}\ell$$

For log-loss: $g_i = p_i - y_i$, $h_i = p_i(1-p_i)$. With $G=\sum g_i$, $H=\sum h_i$ in a leaf, the optimal leaf value is $w^*=-\dfrac{G}{H+\lambda}$ and a split's gain is

$$\tfrac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma .$$

$\lambda$ shrinks leaf values, $\gamma$ is the minimum gain to split, and `min_child_weight` is a floor on $H$ — which is why your Optuna result (`min_child_weight=10`, `reg_alpha≈0.9`) means "regularise heavily".
</details>

---

### 🟡 Q28. LightGBM and CatBoost — what do they change, and did you use them well?

**Topic:** Boosting libraries

<details>
<summary><b>💡 Answer & tips</b></summary>

- **LightGBM:** histogram binning, **leaf-wise** growth, GOSS (keep large-gradient rows) and EFB (bundle exclusive sparse features). Faster, but leaf-wise trees overfit small data without `min_data_in_leaf`/`num_leaves` limits.
- **CatBoost:** symmetric (oblivious) trees, **ordered boosting** and ordered target statistics to avoid prediction shift when handling categoricals.
- **Your usage:** LightGBM F1 0.64 / recall 0.77; CatBoost F1 0.63 / recall 0.75 — both with default-ish settings. CatBoost was fed one-hot dummies, which sidesteps its categorical engine; pass raw categoricals with `cat_features=[...]` to test its real strength. With 7k rows and ≤ 4 levels per column, expect no major gain.
</details>

---

### 🔴 Q29. Where's your logistic-regression baseline? Do you even need boosting?

**Topic:** Baselines · **Interviewer wants:** humility and evidence.

<details>
<summary><b>💡 Answer & tips</b></summary>

The notebook skips it — a gap worth admitting. A repeated-CV re-run (5 folds × 3 repeats, same 34 features):

| Model | ROC-AUC | PR-AUC | Brier |
| :--- | :---: | :---: | :---: |
| Logistic regression (L2, unweighted) | 0.843 ± 0.010 | 0.649 | **0.137** |
| AdaBoost stumps | 0.844 ± 0.010 | 0.659 | 0.241 |
| Gradient boosting | 0.846 ± 0.010 | 0.663 | **0.135** |

**Takeaways:** (1) ranking quality is *saturated* near 0.84 — differences are inside one standard deviation; (2) the AdaBoost Brier score is worse than predicting the base rate (0.195), while logistic regression and gradient boosting are well-calibrated; (3) logistic regression adds coefficient-level interpretability (odds ratios).

**Mature answer:** *"Given equal AUC, I'd now ship calibrated logistic regression or gradient boosting with a threshold from cost. AdaBoost won on my original single-split F1, which is within noise."*
</details>

---

### 🟡 Q30. How did you tune hyperparameters? Compare RandomizedSearchCV and Optuna.

**Topic:** Hyperparameter optimisation

<details>
<summary><b>💡 Answer & tips</b></summary>

- **RandomizedSearchCV:** 50 random draws from a discrete grid, 5-fold CV, `scoring="f1"` → best CV F1 **0.630** (`max_depth=4, lr=0.01, n_estimators=400, scale_pos_weight=2.84`).
- **Optuna (TPE):** 100 trials over continuous/log-uniform ranges → best CV F1 **0.6335** (`max_depth=3, lr≈0.017, n_estimators=627`).
- **TPE idea:** split past trials at a quantile $y^*$ into good/bad, model $\ell(x)=p(x\mid y<y^*)$ and $g(x)=p(x\mid y\ge y^*)$, and propose the $x$ maximising $\ell(x)/g(x)$ (proportional to expected improvement). The first ~10 trials are random; later trials exploit what worked.

**Gap:** the 0.003 CV-F1 difference is far smaller than the fold-to-fold spread (~0.02), so neither method is "better" here.
</details>

---

### 🔴 Q31. What could go wrong in your Optuna study?

**Topic:** Tuning pitfalls

<details>
<summary><b>💡 Answer & tips</b></summary>

1. **Search-space edges:** the best `min_child_weight` is **10 — the upper bound** — and `reg_alpha ≈ 0.91` is near its bound of 1. The optimum is likely outside the range; widen it.
2. **Reproducibility:** `XGBClassifier(**study.best_params)` omits `random_state=42` and `eval_metric` because they were fixed, not tuned.
3. **Winner's curse / multiple comparisons:** after 100 trials the best CV score is optimistically biased. Use **nested CV** or a final untouched hold-out.
4. **Metric choice:** optimising F1 at a fixed 0.5 cut-off ignores the threshold and the business cost; optimise PR-AUC or expected profit instead.
5. **Unused pruning:** Optuna pruners (`MedianPruner`) could cut compute on bad trials.
6. **Leakage:** tuning ran on `x_train` only ✅ — good — but model *selection* later used the test set (audit #2).
</details>

---

### 🟡 Q32. How do you control overfitting in boosting? What is early stopping?

**Topic:** Regularisation

<details>
<summary><b>💡 Answer & tips</b></summary>

Levers: small learning rate with more trees, shallow depth, `min_child_weight`, `gamma`, L1/L2, row/column subsampling, and **early stopping** — monitor a validation metric and stop when it hasn't improved for *k* rounds, so `n_estimators` is learned rather than guessed.

**In this repo:** Optuna fixed 627 trees at `lr≈0.017` without early stopping — a lower-variance choice would be `n_estimators=2000` with `early_stopping_rounds=50` on a validation fold. For AdaBoost, `learning_rate < 1` plus fewer stumps plays the same role; yours uses `learning_rate=1.0`.

**Diagnostic:** plot train vs validation loss against rounds; a widening gap means overfitting.
</details>

---

### 🔴 Q33. How would you prove model A is really better than model B?

**Topic:** Statistical rigour

<details>
<summary><b>💡 Answer & tips</b></summary>

- Use **repeated stratified k-fold** with *identical folds* for both models, then a **paired** test on the per-fold scores (or a bootstrap CI on the difference).
- For AUC use DeLong or bootstrap; for paired predictions at one threshold use **McNemar**.
- Tuning claims need **nested CV** (inner loop tunes, outer loop evaluates).
- Rule of thumb from your data: SE of recall ≈ 0.02 on a 1,407-row split and AUC std ≈ 0.01 across folds, so gaps < 0.02 are indistinguishable.

**Say:** *"I'd only call a model better if the paired CI excludes zero and the gain matters in dollars."*
</details>

---

### 🟡 Q34. Random Forest vs boosting — bias, variance and your results.

**Topic:** Ensembles

<details>
<summary><b>💡 Answer & tips</b></summary>

- **Bagging (RF):** averages many deep, decorrelated trees → mostly **variance** reduction.
- **Boosting:** adds shallow learners sequentially → mostly **bias** reduction.
- **Your numbers:** RF (`class_weight="balanced"`, depth 6, 300 trees) → precision 0.51 / recall 0.80 / F1 0.62; RF + SMOTEENN (500 trees) → 0.51 / 0.82 / 0.63. Comparable to boosting.
- **Doc inconsistency to fix:** the README table lists RF as 0.59 / 0.65 / 0.54 — it does not match the notebook output above (audit #10).
- RF probabilities are the average of leaf frequencies and are usually better calibrated than AdaBoost's, but still benefit from calibration.
</details>

---

# Part 4 — Evaluation, Calibration, Explainability & Business (Q35–Q42)

### 🟡 Q35. Read your confusion matrix and put a price on each cell.

**Topic:** Metrics → money

<details>
<summary><b>💡 Answer & tips</b></summary>

Notebook (n = 1,407): **TN 706 · FP 297 · FN 71 · TP 333**.

- Recall = 333 / 404 = **0.82** (we reach 82% of churners).
- Precision = 333 / 630 = **0.53** (about half of the people we contact would have stayed anyway).
- Specificity = 706 / 1,003 = 0.70.

| Cell | Cost |
| :--- | :--- |
| **FN** (missed churner) | lose customer value or replacement cost (app assumes $1,200) |
| **FP** (needless offer) | offer cost + possible discount leakage (e.g. $50) |
| **TP** | offer cost, but customer saved with probability *s* |

With FN ≫ FP the weighted model's low precision is acceptable. State this *before* the interviewer asks "why is precision only 53%?"
</details>

---

### 🟡 Q36. PR-AUC vs ROC-AUC — which do you report, and what do your numbers say?

**Topic:** Metrics

<details>
<summary><b>💡 Answer & tips</b></summary>

- **ROC-AUC** = probability a random churner outranks a random non-churner; insensitive to prevalence, so it looks flattering under imbalance.
- **PR-AUC** (average precision) is anchored to prevalence — the no-skill baseline is **0.265** — and directly reflects the precision you'll get when contacting customers.
- **Re-run (10 stratified seeds):** ROC-AUC **0.844 ± 0.008**, PR-AUC **0.656 ± 0.012** (2.5× the baseline).

**Note on the app:** the sidebar's ROC-AUC 0.842 is plausible and reproducible; its "81.2% accuracy" is not (≈74.6% for the weighted model). Metrics shown in a UI should be loaded from a stored evaluation artifact.
</details>

---

### 🔴 Q37. How do you pick the threshold? Derive it from costs.

**Topic:** Decision theory

<details>
<summary><b>💡 Answer & tips</b></summary>

For a **calibrated** probability $p$, contact the customer if the expected gain is positive:

$$p\cdot s\cdot V - c > 0 \;\Longrightarrow\; p > \frac{c}{s\,V}.$$

With $c=\$50$, $s=0.30$, $V=\$1{,}200$ the cut-off is $\approx 14\%$. If you ignore $s$ and treat it as pure cost-sensitive classification, the cut-off is $t^*=\dfrac{C_{FP}}{C_{FP}+C_{FN}}$.

**Practical recipe:** (1) calibrate; (2) sweep thresholds on a validation set; (3) maximise expected profit subject to a **campaign capacity** (e.g. call centre can handle top 10% only); (4) re-check after every retrain.

**Warning:** this only works if scores are probabilities. A cut-off of 0.45/0.50 on the deployed raw scores is an *ad-hoc* rank cut, not a cost-optimal rule (Q38).
</details>

---

### 🔴 Q38. Are the app's "churn probabilities" real probabilities?

**Topic:** Calibration · **Interviewer wants:** you noticed. This is the most valuable question in the guide.

<details>
<summary><b>💡 Answer & tips</b></summary>

**No.** Run the deployed pickle exactly as `app.py` does:

- The three built-in personas score **51.1%** (high-risk), **50.0%** (moderate) and **48.0%** (loyal).
- Across all customers the 1st–99th percentile of scores is only **0.474–0.515**.
- Brier score 0.244 vs 0.195 for a constant 26.5% forecast.

Yet the **ranking is strong** (held-out, 5 score quintiles):

| Score range | n | Mean score | Observed churn |
| :--- | :---: | :---: | :---: |
| 0.31 – 0.486 | 282 | 0.480 | **0.7%** |
| 0.486 – 0.494 | 281 | 0.490 | 7.1% |
| 0.494 – 0.500 | 281 | 0.497 | 19.2% |
| 0.500 – 0.505 | 281 | 0.503 | 39.9% |
| 0.505 – 0.520 | 282 | 0.509 | **66.0%** |

The top quintile has 2.5× lift and ≈50% of all churners; the top two quintiles hold ≈80%.

**Consequences in the app:** "Moderate Risk (45–50%)" actually churns at ~9%, not "significantly above the 26.5% baseline"; the README's 35%/60% tiers can never trigger; "+24.6% vs baseline" for a 51% score is meaningless.

**Fix:** wrap the model in `CalibratedClassifierCV(method="isotonic", cv=5)` (or `"sigmoid"` for small data), check a reliability curve and Brier score, derive tiers from calibrated probabilities and expected cost — or ship logistic regression / gradient boosting, which are calibrated out of the box (Brier ≈ 0.135).
</details>

---

### 🔴 Q39. How do you explain a prediction? Is the app's feature chart an explanation?

**Topic:** Explainability

<details>
<summary><b>💡 Answer & tips</b></summary>

**The app's chart is static**: a hard-coded copy of the model's global importances (`TotalCharges` 0.36, `MonthlyCharges` 0.26, contracts 0.06 each, …) identical for every customer. The retention playbook below it is rule-based (`if Contract == "Month-to-month": …`), not model-derived.

**Better:**
- **SHAP** — additive per-feature contributions that sum to the prediction (Shapley values: fair credit allocation). Use `TreeExplainer` for XGBoost/GBM; the model-agnostic permutation/kernel explainers work for AdaBoost.
- **Why not impurity importance?** It is biased toward continuous/high-cardinality features: `TotalCharges` + `MonthlyCharges` hold 62% simply because they offer more split points. Compare with **permutation importance** on held-out data.
- For linear models, coefficients/odds ratios give global *and* local explanations for free.
</details>

---

### 🟡 Q40. What drives churn, and which findings are actionable versus just correlation?

**Topic:** Insight → action

<details>
<summary><b>💡 Answer & tips</b></summary>

- **Drivers:** contract type, tenure, price level (monthly charges), fiber without support, electronic-check payment.
- **Actionable levers:** term-contract incentives, autopay credits, bundling tech support/security, onboarding calls in months 1–12.
- **Correlation trap:** *"Eliminate month-to-month contracts"* would not remove the *kind of customer* who chooses month-to-month (price-sensitive, uncertain). Treat features as risk markers, not causes.
- Only **randomised experiments** (Q41) tell you whether an incentive changes behaviour.
</details>

---

### 🔴 Q41. What is uplift modelling, and how would you validate retention ROI?

**Topic:** Causal ML

<details>
<summary><b>💡 Answer & tips</b></summary>

A churn model finds *who will leave*; an **uplift model** finds *whom a treatment will change*. Customers fall into four groups: **persuadables** (target), sure things, lost causes, and *sleeping dogs* (contact makes them leave).

- **Data:** randomised treated/control campaign. **Models:** two-model (T-learner), S-learner, or causal forests; **metrics:** Qini curve / AUUC.
- **Feedback-loop fix:** keep a 5–10% **universal holdout** that gets no offer. It gives (a) clean labels for retraining — otherwise saved customers are labelled "stayed" and the model unlearns the risk signal — and (b) the true campaign ROI.
</details>

---

### 🟡 Q42. Fairness, privacy and compliance — anything to worry about?

**Topic:** Responsible ML

<details>
<summary><b>💡 Answer & tips</b></summary>

- **Protected/proxy attributes:** `gender` and `SeniorCitizen` (age) are inputs. Churn risk differing by age is not itself unlawful, but offering better deals only to high-risk groups can create **disparate impact**. Audit error rates and selection rates by group (equal opportunity, demographic parity gap); consider dropping `gender`.
- **Privacy:** real billing data is personal data — minimise columns, restrict access, log predictions without raw PII, define retention.
- **Transparency:** if decisions affect pricing or service, keep reason codes (SHAP) available.
- **Repo hygiene:** the public dataset is anonymised sample data; never commit real customer files.
</details>

---
# Part 5 — Production, MLOps & System Design (Q43–Q50)

### 🟡 Q43. Is pickling the model safe? How do you version and pin it?

**Topic:** Model artifacts

<details>
<summary><b>💡 Answer & tips</b></summary>

- **Version coupling:** the pickle was written by scikit-learn **1.5.0**, but `requirements.txt` allows `>=1.4.0,<1.6.0`. Loading it under 1.5.2 prints one "Trying to unpickle estimator … from version 1.5.0 when using version 1.5.2" warning *per tree* (51 in total). Pin the exact version: `scikit-learn==1.5.0`, and use the same Python/NumPy in Docker, Streamlit Cloud and CI.
- **Security:** `joblib.load` / `pickle.load` can execute arbitrary code. Only load artifacts you built; never load user-supplied files; verify a `sha256` checksum in CI.
- **Safer or portable formats:** `skops.io` (restricted types), ONNX (`skl2onnx`) for cross-language serving, or a model registry (MLflow) with metadata.
- **Always store beside the model:** training sklearn version, git SHA, data hash, feature list, evaluation metrics, decision threshold — in a `model_card.json`.
</details>

---

### 🟡 Q44. Explain Streamlit's execution model and the caching in `app.py`.

**Topic:** App architecture

<details>
<summary><b>💡 Answer & tips</b></summary>

- Streamlit **re-runs the whole script** on every widget interaction.
- `@st.cache_data` caches *serialisable* results and returns a copy — right for the CSV.
- `@st.cache_resource` caches *global* objects (models, connections) shared across **all sessions and threads** — so they must be treated as read-only.
- **Improvements for this app:**
  1. `build_preprocessor(df)` takes a DataFrame argument, so Streamlit hashes it on every rerun; use `_df` to skip hashing or build it inside `load_data`.
  2. The results panel is guarded by `if predict_clicked:`; touching *any* widget re-runs the script with the button back to `False`, so the report vanishes. Store the result in `st.session_state`.
  3. Heavy work runs only on click ✅ — keep it that way.
</details>

---

### 🔴 Q45. Walk me through your Dockerfile's security and caching — and what would you still change?

**Topic:** Containers

<details>
<summary><b>💡 Answer & tips</b></summary>

**Already good:** `python:3.11-slim`, non-root `appuser` (UID 10001), `requirements.txt` copied before code (dependency layer cached), `PYTHONDONTWRITEBYTECODE`, `HEALTHCHECK` on `/_stcore/health`, `.dockerignore` excludes notebooks and secrets.

**Be ready to admit and fix:**
- It is **single-stage**; the README calls it "multi-stage". A multi-stage build only pays off if you compile wheels — otherwise say "slim, single-stage".
- `COPY . .` also ships `presentation/` (PPTX + PDF), `production/`, `Interview Questions/` and two unused model pickles → use explicit `COPY app.py Data/ Models/Ada_boost_churn_model.pkl`.
- `useradd -s /bin/bash` → use `/usr/sbin/nologin`; healthcheck can use Python's `urllib` and drop `curl` to shrink the attack surface.
- Pin the base image by **digest**, scan with Trivy / Docker Scout, tag images with the git SHA (not only `latest`).
- Runtime flags: `--read-only --tmpfs /tmp --cap-drop=ALL --security-opt no-new-privileges:true --memory 1g`.

Layer cache in one line: Docker reuses a layer if its instruction and all inputs are unchanged, so rarely-changing steps go first.
</details>

---

### 🔴 Q46. Streamlit Cloud, EC2, ECS Fargate — how do you choose where to deploy?

**Topic:** Deployment architecture

<details>
<summary><b>💡 Answer & tips</b></summary>

| Option | Best for | Watch out |
| :--- | :--- | :--- |
| **Streamlit Community Cloud** | Free portfolio demo, GitHub push-to-deploy | Apps with no traffic for 12 hours hibernate; resource limits can change without notice |
| **Docker on one VM (EC2 + Nginx + Certbot)** | Cheap, full control | You patch the OS, handle TLS renewal and restarts; Nginx must forward WebSocket upgrade headers |
| **ECS on Fargate + ALB** | Production, HA, autoscaling | More moving parts; enable ALB **sticky sessions** with >1 task because Streamlit holds session state over a WebSocket |
| **ECS Express Mode** | Fastest managed path on AWS — provisions Fargate, ALB/HTTPS and scaling from an image and two IAM roles | Newer; still verify health-check path and port |
| ~~AWS App Runner~~ | — | **Closed to new customers since 30 Apr 2026** (existing customers unaffected); AWS points to ECS Express Mode |

**Rule:** demo → Streamlit Cloud; internal tool → one VM or Express Mode; customer-facing → Fargate/ALB with private subnets, WAF, alarms. Detailed runbooks live in [`production/`](../production/).
</details>

---

### 🔴 Q47. How do you monitor this model after launch when churn labels arrive 1–3 months late?

**Topic:** Drift & monitoring

<details>
<summary><b>💡 Answer & tips</b></summary>

1. **Data/feature drift (immediate):** per-feature PSI (rule of thumb: < 0.1 stable, 0.1–0.25 watch, > 0.25 act), KS test for numeric, χ² for categorical; also schema/range checks and null rates.
2. **Prediction drift (immediate):** monitor *rank-based* statistics — share of customers in the top decile, score quantiles. Because raw AdaBoost scores live in a 0.47–0.52 band (Q38), absolute-threshold alerts are meaningless until calibrated.
3. **Performance (delayed):** once outcomes mature, compute recall/precision/lift **per scoring cohort**; compare with the training baseline (recall ≈ 0.80, PR-AUC ≈ 0.66).
4. **Business KPIs:** save rate, campaign ROI, holdout churn.
5. **Concept drift triggers:** new pricing, competitor launches, network incidents. **Tools:** Evidently, WhyLogs, Prometheus + Grafana, CloudWatch alarms.
</details>

---

### 🔴 Q48. How do you retrain and release a new version safely?

**Topic:** MLOps lifecycle

<details>
<summary><b>💡 Answer & tips</b></summary>

- **Triggers:** schedule (monthly/quarterly) *plus* drift or performance alerts.
- **Gate:** the challenger must beat the champion on a fixed validation set with a **paired CI** (Q33) *and* pass calibration/Brier checks, a parity test (Q49) and a latency budget.
- **Rollout:** shadow mode (score, don't act) → canary (5–10% traffic) → full; roll back by redeploying the previous image tag.
- **Feedback loop:** keep the universal holdout (Q41) so retraining labels aren't contaminated by retention actions.
- **Reproducibility:** seed everything, version data (DVC/hash), log runs to MLflow, and write the metrics file the app reads (replacing hard-coded sidebar numbers).
</details>

---

### 🟡 Q49. How would you test and CI/CD this project?

**Topic:** Software quality for ML

<details>
<summary><b>💡 Answer & tips</b></summary>

**Unit tests (pytest)** for the pieces that silently break: tenure bin edges (`0 → "1-12"`, `12 → "1-12"`, `13 → "13-24"`), output shape `(1, 34)`, unseen category → error, month-to-month scores above two-year for otherwise identical customers, and a **parity test** (notebook pipeline vs app pipeline on 100 held-out rows).

```python
def test_tenure_bins():
    from app import preprocess_input  # refactor pure functions out of Streamlit
    x, _ = preprocess_input(sample(tenure=12), prep)
    assert x.shape == (1, 34)
```

**GitHub Actions:** lint (ruff) → tests → `docker build` → Trivy scan → push to ECR with **OIDC** (no stored AWS keys) → deploy. Add `pre-commit`, pinned requirements and a model-card check.
</details>

---

### 🔴 Q50. How would you evolve this prototype into an enterprise service?

**Topic:** System design

<details>
<summary><b>💡 Answer & tips</b></summary>

**First ask:** *real-time or batch?* Churn outreach is usually **weekly batch** — real-time scoring is only needed for agent-assist during a live call.

- **Batch path:** warehouse → feature job → scoring job (Spark/Ray or plain pandas) → write scores to CRM. The model scores **10,000 rows in ≈ 60 ms** (≈ 0.006 ms/row) on a laptop-class CPU.
- **Online path:** FastAPI + Pydantic validation, `/predict` and `/health`, model loaded once per worker, autoscaled behind a load balancer. A single-row call measures **≈ 4.6 ms p50 (model only)** — so the app's "sub-millisecond" caption is inaccurate; measure p50/p95 with a load test.
- **Platform:** feature store for point-in-time features, model registry, drift monitors, A/B holdout, audit log of scores.
- **Security:** authentication, rate limits, TLS, secrets in AWS Secrets Manager, least-privilege IAM, no PII in logs.
- **Separation of concerns:** UI (Streamlit) ↔ API ↔ model service, so each scales and deploys independently.
</details>

---

# ⚡ Rapid-Fire Round

<details>
<summary><b>💡 25 one-line answers to practise out loud</b></summary>

1. **Churn rate?** 26.54% (1,869 of 7,043).
2. **Why not accuracy?** "Always No" already scores 73.4%.
3. **Blank `TotalCharges`?** 11 rows, all `tenure = 0`; coerced and dropped.
4. **Features after encoding?** 34.
5. **Why `drop_first`?** Avoid perfectly collinear dummies; harmless for trees.
6. **Deployed model?** AdaBoost, 50 stumps, `sample_weight = 2.84`.
7. **Recall / precision?** ≈ 0.80 / 0.51 (10-seed re-run); 0.82 / 0.53 on the notebook split.
8. **ROC-AUC?** 0.844 ± 0.008.
9. **Is 0.64 vs 0.63 F1 a real difference?** No — noise is ±0.02.
10. **Does AdaBoost need scaling?** No.
11. **Can stumps model interactions?** No — additive; depth-2 AdaBoost was *worse* (AUC 0.822).
12. **Are the scores probabilities?** No; they sit in 0.47–0.52 and need calibration.
13. **Best simple baseline?** Logistic regression — AUC 0.843, Brier 0.137.
14. **Threshold rule?** Contact if $p > c/(sV)$ using calibrated $p$.
15. **SMOTE risk here?** Interpolating one-hot dummies yields non-real customers.
16. **Where to resample?** After the split, inside each CV fold.
17. **Top churn driver?** Contract type (42.7% vs 2.8%), then tenure.
18. **Fiber without support?** 49.4% churn vs 22.6% with support.
19. **Static feature chart?** Global importance, not per-customer.
20. **Pickle risk?** Arbitrary code execution; pin versions, checksum, prefer skops/ONNX.
21. **Docker security?** Non-root user, health check, slim base, read-only FS, capability drop.
22. **Streamlit Cloud sleep?** After 12 hours without traffic.
23. **App Runner?** Closed to new customers since 30 Apr 2026 → ECS Express Mode.
24. **Feedback loop fix?** Universal holdout group.
25. **Biggest next step?** Calibrated model + Pipeline artifact + parity tests.
</details>

---

# 🧪 Reproduce the Audit Numbers

<details>
<summary><b>💡 Script that regenerates the re-run metrics (scikit-learn 1.5.x)</b></summary>

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             recall_score, precision_score, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Data/Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["tenure_bin"] = pd.cut(df["tenure"], [0, 12, 24, 36, 48, 60, 72],
                          labels=["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"], include_lowest=True)
y = (df["Churn"] == "Yes").astype(int)
X = pd.get_dummies(df.drop(columns=["customerID", "Churn", "tenure"]), drop_first=True)

rows = []
for seed in range(10):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    w = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
    clf = AdaBoostClassifier(n_estimators=50, random_state=42).fit(sc.transform(Xtr), ytr, sample_weight=w)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]; pred = (p >= 0.5).astype(int)
    rows.append([accuracy_score(yte, pred), roc_auc_score(yte, p), average_precision_score(yte, p),
                 recall_score(yte, pred), precision_score(yte, pred), f1_score(yte, pred), brier_score_loss(yte, p)])

print(pd.DataFrame(rows, columns=["acc", "roc_auc", "pr_auc", "recall", "precision", "f1", "brier"])
      .agg(["mean", "std"]).round(3))
```
</details>

---

*Curated for the Telecom Customer Churn Prediction project · numbers verified 28 Oct 2026 · answers are written to be spoken, not memorised.*
