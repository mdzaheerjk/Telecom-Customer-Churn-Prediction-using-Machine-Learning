# 📱 Telecom Churn Prediction — Production-Grade Interview Questions

50 questions (with answers) focused on production-readiness, MLOps, and system design for this churn prediction project (EDA → imbalanced classification → Streamlit deployment).

---

## 🚨 Section 1: Training–Serving Skew & Preprocessing Reproducibility (Q1–Q10)

**1. There's a critical bug in how the Streamlit app builds its `StandardScaler`. What is it?**
`build_preprocessor()` fits a *brand-new* `StandardScaler` on the entire raw dataset at app startup, but the deployed model (`Ada_boost_churn_model.pkl`) was trained on features scaled by a *different* `StandardScaler` — one fit only on the training split (`x_train`) inside the modeling notebook. These are two different fitted scalers with different means/variances, so every prediction served in production is being made on inputs scaled inconsistently with how the model was trained.

**2. Why wasn't this scaler mismatch caught by testing?**
Because the app "works" — it runs, takes input, returns a prediction — with no error thrown. Scaling mismatches are silent: they don't crash, they just quietly bias every prediction. This is exactly the class of bug that unit tests checking "does it run" miss, and only tests that check "are the values numerically correct against the training pipeline" would catch.

**3. What's the fix, architecturally?**
Persist the *exact* fitted preprocessing objects (`StandardScaler`, and the one-hot column template) from the training notebook via `joblib.dump`, alongside the model, and load all three artifacts together at inference time — never refit any transformer inside the serving app.

**4. Why is fitting a `ColumnTransformer`/pipeline (as in the loan project) generally safer here than the manual `pd.get_dummies` + separate `StandardScaler` approach used in this project?**
A single fitted `Pipeline` object bundles preprocessing and model together and is serialized as one artifact — there's no way to accidentally "forget" a step or refit a component at serving time, because loading the pipeline loads the exact fitted transformers used in training.

**5. `pd.get_dummies(df_in, drop_first=True)` is called on a *single-row* input DataFrame at inference time. What's the risk?**
One-hot encoding a single row can only produce dummy columns for the categories present in *that* row — if the applicant's category was the one dropped as baseline (`drop_first=True`), or if a category never appears, the resulting columns won't match the training-time column set at all without the `reindex(columns=template_cols, fill_value=0)` step. The code does reindex correctly here, which is good, but this only works because the template columns were captured — it's fragile and easy to break if that step is ever removed or the template drifts from what the model actually expects.

**6. What is "training-serving skew" and where else does it show up in this project besides the scaler?**
It's when the data or transformations seen at inference differ from what the model was trained on. Here it also shows up in `tenure_bin`: the app rebinds `tenure` into bins using the bins/labels captured from the *app's own* `build_preprocessor` call — not necessarily identical to how `tenure_bin` was constructed in the original training notebook (the bins/labels values match, but they're independently re-declared in two files, meaning a future edit to one and not the other silently breaks parity).

**7. The app fits `build_preprocessor` on `Customer-Churn.csv` loaded fresh every deploy (cached via `st.cache_resource`). Why is fitting preprocessing artifacts inside the serving app an anti-pattern?**
It couples serving to having the *exact same* training dataset available and unchanged at deploy time. If the CSV is ever updated, re-processed, or has even one row added/removed, the scaler and dummy-column template silently change — meaning the exact same input could get a different prediction after a routine data refresh, with no version bump or warning.

**8. What would a proper ML pipeline artifact bundle look like for this project?**
A single versioned bundle (e.g. `model_v3/`) containing: the fitted preprocessing pipeline, the fitted model, a `feature_manifest.json` listing expected columns in order, a training data/code commit hash, and evaluation metrics from that training run — loaded together, never reconstructed at serving time.

**9. How would you unit test that the serving preprocessing exactly matches training preprocessing?**
Snapshot a few real rows' fully-preprocessed feature vectors at the end of the training notebook, save them as a fixture, then add a test that runs those same raw rows through the serving app's `preprocessor_input()` and asserts the output matches the fixture within floating-point tolerance.

**10. Debug `print()` statements are scattered through `load_data`, `build_preprocessor`, `preprocessor_input`, and `load_model`. Why is this a production concern beyond just "messy code"?**
In a real deployment `print()` output typically goes to stdout logs with no structure, level, or context (no request ID, timestamp format, or severity) — making it nearly useless for debugging at scale and impossible to filter or alert on. It should be replaced with a structured logger (e.g. Python's `logging` module) using appropriate levels (`DEBUG`, `INFO`, `WARNING`).

---

## ⚖️ Section 2: Class Imbalance in Production (Q11–Q20)

**11. The dataset is ~73:27 imbalanced. Why does the app report a raw prediction and probability without addressing this?**
The deployed AdaBoost model was trained with `sample_weight` weighting to compensate for imbalance during training itself, so at inference time no additional rebalancing is needed — the model's `predict`/`predict_proba` already reflect the weighted training objective. This is worth confirming explicitly in code comments, since it's not obvious from reading the app alone.

**12. Why is accuracy a misleading metric for this problem, concretely?**
Always predicting "No churn" gets ~73% accuracy for free, since that's the majority class — a model needs to be evaluated on precision/recall/F1 for the minority (churn) class specifically to know if it's adding any real value.

**13. Compare SMOTEENN, SMOTE, ADASYN, and class-weighting as imbalance strategies used in this project — what are the production trade-offs, not just accuracy?**
SMOTE/SMOTEENN/ADASYN synthesize new training rows, which means retraining pipelines need the `imblearn` dependency and extra preprocessing steps that only apply during training (never at inference) — an easy place to accidentally leak resampling into a serving pipeline. Class-weighting (`class_weight='balanced'`, `scale_pos_weight`, `sample_weight`) needs no extra dependency at inference and is simpler to reproduce, but requires the algorithm to support weighting natively.

**14. Several models were resampled with `sm.fit_resample(x_train, y_train)` — was this correctly applied only to the training set, or is there a leakage risk?**
Correctly applied to `x_train`/`y_train` only, with `x_test`/`y_test` left untouched — this is the right pattern. Applying SMOTE before the train/test split would leak synthetic points derived from test-set neighbors into training, artificially inflating test performance.

**15. `RandomizedSearchCV`/Optuna both optimize `scoring='f1'` (or an F1 objective) via cross-validation on `x_train`. Is there a subtle imbalance-related leakage risk in how CV was set up?**
Yes, potentially — plain `cross_val_score`/`RandomizedSearchCV` with default CV doesn't guarantee each fold preserves the class ratio unless `StratifiedKFold` is used (which is `cv`'s default behavior for classifiers in scikit-learn, so likely fine here) — but if any resampling were applied *before* the CV split rather than inside each fold, that would leak synthetic minority samples across folds. Here, resampling happens outside CV objects entirely (manual `fit_resample` before fitting), so it's fine for those specific runs, but worth being explicit about for anyone extending this.

**16. In production, which error is more costly for a telecom company: false positive (predicting churn for a loyal customer) or false negative (missing a customer who will churn)?**
Typically false negatives are costlier — a missed churner represents lost recurring revenue with no retention intervention, whereas a false positive just costs an unnecessary (and often cheap) retention offer to a loyal customer. This argues for optimizing recall on the churn class, consistent with the project's choice to weight/resample toward catching more churners (recall for class 1 reached ~0.80–0.82 in several models, at the cost of precision).

**17. Given that recall/precision trade-off, how would you choose an operating threshold in production rather than using the default 0.5 cutoff?**
Plot a precision-recall curve on a held-out validation set, then pick the threshold that hits a business-defined minimum recall (e.g. "catch at least 75% of churners") while maximizing precision at that recall level — and revisit the threshold periodically as the underlying churn base rate or retention program economics change.

**18. Why might it be dangerous to compare F1 scores across models trained with *different* imbalance-handling techniques (SMOTEENN's 0.61 vs class-weighted AdaBoost's 0.64 vs Optuna-tuned XGBoost's 0.63)?**
Because F1 computed on differently-processed test sets, different random splits (note: no `random_state` was set on any of the `train_test_split` calls in the modeling notebook!), or slightly different pipelines isn't a perfectly like-for-like comparison — small differences could be due to random split variance rather than real model superiority.

**19. Speaking of which — `train_test_split(x, y, test_size=0.2)` is called multiple times across the notebook *without* a `random_state`. Why is this a serious reproducibility problem?**
Every rerun of the notebook produces a different train/test split, so reported metrics aren't reproducible run-to-run, and "best model" comparisons between cells (each implicitly re-splitting or reusing stale splits) may not even be evaluated on the same test set — undermining the validity of the whole model comparison exercise.

**20. How would you validate that a chosen imbalance strategy actually generalizes, rather than overfitting to this particular test split?**
Use repeated stratified k-fold cross-validation (not a single train/test split) and report the mean and standard deviation of F1/recall/precision across folds — a strategy that looks best on one lucky split may not hold up, and the variance itself is informative about how much to trust the comparison.

---

## 🧪 Section 3: Model Selection & Experiment Governance (Q21–Q28)

**21. Three different "best" models get saved in this project (`best_xgboost_churn_model.pkl`, `Best_optuna_churn_model.pkl`, `Ada_boost_churn_model.pkl`), but the app hardcodes loading the AdaBoost one. Why is this a red flag?**
There's no clear, documented decision trail for *why* AdaBoost was chosen for deployment over the Optuna-tuned XGBoost model, which had comparable or arguably better CV F1 (~0.634 vs untuned AdaBoost's test F1 ~0.64, but on different splits) — this looks like an ad hoc choice rather than a governed model-selection decision, which is a real risk in production ML.

**22. What tool/practice would you introduce to fix this governance gap?**
An experiment tracker (e.g. MLflow, Weights & Biases) logging every run's hyperparameters, metrics, data version, and artifact — with a clear "promote to production" step that records which run was selected, by whom, and why, instead of models being saved ad hoc as loose `.pkl` files with inconsistent naming (`best_xgboost_churn_model.pkl` vs `Ada_boost_churn_model.pkl` — inconsistent casing alone suggests no naming convention).

**23. The Optuna study ran 100 trials optimizing F1 via 5-fold CV. What's a risk of trusting the single best trial's hyperparameters without further validation?**
With 100 trials all evaluated via stochastic CV, some amount of the "best" score is due to noise/overfitting to the CV folds themselves (multiple-comparisons effect) — the reported best CV F1 (0.634) may not hold up as well on a fresh, truly unseen test set, so it's worth validating the final chosen config once more on a separate holdout never used during search.

**24. Why fit both `RandomizedSearchCV` (50 iterations) *and* a separate Optuna study (100 trials) on essentially the same search space for XGBoost? What would you do differently to make this more efficient?**
This duplicates search effort with two different search strategies instead of committing to one well-configured approach — Optuna's Bayesian/TPE search is generally more sample-efficient than random search, so running `RandomizedSearchCV` first likely added compute cost without adding much information beyond what Optuna alone would find, especially with 100 trials.

**25. `CatBoostClassifier` was fit with `verbose=0` — what production consideration does the verbosity setting reflect, and what's a downside of silencing it entirely?**
Silencing training logs keeps notebook output clean, but in a production training pipeline you generally want at least periodic loss/metric logging captured into your experiment tracker or logs — fully silencing it means losing visibility into whether training converged smoothly or, e.g., hit early stopping/overfit partway through.

**26. Several models (RF, XGBoost, CatBoost, LightGBM, AdaBoost) were tried with minimal or no hyperparameter tuning for most of them, while XGBoost alone got two rounds of extensive tuning. Is this a fair model comparison?**
No — comparing a heavily-tuned XGBoost against untuned or lightly-tuned RF/CatBoost/LightGBM biases the comparison in XGBoost's favor. A fair bake-off would budget comparable tuning effort (e.g. the same number of Optuna trials) across all candidate algorithms before picking a winner.

**27. What would a proper "model card" for the deployed AdaBoost model include, and why does this project lack one?**
Training data description and time period, feature list, preprocessing steps, evaluation metrics with confidence intervals, known limitations (e.g. imbalance-adjusted metrics, no external validation set), and intended use/misuse boundaries. This project has none of that documentation — anyone inheriting this codebase would have to reverse-engineer the modeling notebook to understand what's actually running in production.

**28. If this churn model were retrained monthly on fresh data, what would you add to catch a case where the new model is *worse* than the currently deployed one?**
A gate in the retraining pipeline that evaluates the new candidate against a fixed, held-out "golden" validation set (not just the newest data's own split) and blocks promotion to production unless it meets or beats the current model's metrics on that fixed benchmark — otherwise silent regressions can ship unnoticed.

---

## 🖥️ Section 4: Streamlit App Architecture (Q29–Q36)

**29. Why does `build_preprocessor` use `@st.cache_resource` while `load_data` uses `@st.cache_data`? What's the actual difference?**
`st.cache_data` is for data that Streamlit can safely hash and serialize (DataFrames, arrays) — it may recompute if the underlying data changes. `st.cache_resource` is for non-serializable objects like fitted models/scalers/DB connections that should be created once and reused as a singleton across reruns — using the wrong decorator (e.g. `cache_data` on a fitted `StandardScaler` object) can silently fail to cache correctly or throw serialization errors.

**30. `min_value`, `max_value`, and `value` for `st.number_input` are all computed from the *training* sample data's min/max/median. What happens to a real-world applicant whose true value falls outside that range?**
`st.number_input`'s `min_value`/`max_value` hard-clamp the widget — a user literally cannot enter a value outside the training data's observed range, silently constraining real-world applicability to whatever range happened to exist in the training set (e.g. if training data's max `MonthlyCharges` was $118.75, no one can be scored above that even if real charges have since increased).

**31. Is there a way for the model to be scored with a completely different set of dummy columns than what it saw during training, given how `cols_to_ask` is dynamically built from `sample.columns`?**
Potentially — `cols_to_ask` is derived from whatever columns exist in the *currently loaded* CSV at runtime, not from a fixed manifest of the columns the deployed model was actually trained on. If the CSV schema changes (a column renamed/added/removed), the UI and the reindexed `template_columns` would silently diverge from what the model expects, only saved by the `reindex(fill_value=0)` masking the mismatch as zeros rather than an error.

**32. Why use `st.form` to wrap all the inputs and a single submit button, rather than triggering prediction on every widget change?**
Without `st.form`, every single widget interaction (each slider drag, each dropdown change) triggers a full script rerun and, without careful caching, potentially a new prediction call — `st.form` batches all inputs and only submits (and predicts) once the user clicks "Predict," which is both a better UX and avoids wasted compute.

**33. The app builds the entire feature-collection UI dynamically by looping over `sample.columns`. What's a maintainability risk of this pattern versus explicitly defining each input field?**
It's fragile to any change in the underlying CSV — adding an unrelated new column to `Customer-Churn.csv` would automatically add a new (unvetted, unlabeled-for-UX) input field to the production UI with no code review of how that field should be validated, labeled, or explained to the end user.

**34. What load/performance concern exists with re-running `build_preprocessor(df)` (which does `pd.get_dummies` on the *entire* dataset) on every cache miss?**
For this dataset size (7K rows) it's cheap, but this pattern doesn't scale — if the training data grew substantially, refitting a scaler and rebuilding the dummy-encoding template over the full dataset on every app cold-start (or cache invalidation) becomes a real startup-latency and memory concern, especially in a multi-instance deployment where every instance repeats this work independently.

**35. The "Debug Information" expander shows raw input, reindexed features, and scaled values directly to end users. Is this appropriate for production?**
Generally no — exposing internal feature engineering and preprocessing details to end users is an information-disclosure risk (reveals exactly how the model works, easing adversarial gaming of the model) and clutters the UX for non-technical users; this belongs behind a feature flag, admin-only view, or should be removed and replaced with proper backend logging instead.

**36. How would you add input validation to catch nonsensical combinations (e.g. `PhoneService = No` but `MultipleLines = Yes`) that the training data would never contain?**
Add cross-field validation rules mirroring the dataset's logical constraints (e.g. if `PhoneService == 'No'`, force `MultipleLines` to `'No phone service'` and disable/hide the field) before submission — right now the UI allows building a combination that was structurally impossible in training data, which the model has never learned to handle sensibly.

---

## 🚀 Section 5: Deployment, Monitoring & Reliability (Q37–Q46)

**37. `joblib.load(path)` is called directly on the model path with no validation. What security risk does this carry in a production setting?**
`joblib.load` (like `pickle`) can execute arbitrary code embedded in the file during deserialization — if the model artifact's source or storage location isn't fully trusted/access-controlled, loading it is a code-execution risk, not just a data-loading operation. Model artifacts should come from a signed/verified internal registry, never an unauthenticated or user-writable path.

**38. Neither the Streamlit app nor the training notebooks pin exact library versions (scikit-learn, XGBoost, CatBoost, LightGBM, imbalanced-learn, joblib). Why does this matter specifically for `.pkl` model files?**
Pickled scikit-learn-family models can be sensitive to the exact library version they were serialized with — loading a model pickled under one sklearn version with a different version installed at serving time can raise warnings, silently change behavior, or fail outright. A `requirements.txt`/lockfile pinning exact versions used at training time is essential for reproducible deployment.

**39. How would you monitor this deployed model for data drift once it's live?**
Track the distribution of key inputs (e.g. mean/std of `MonthlyCharges`, category frequencies for `Contract`, `PaymentMethod`) over time against the training distribution, using a statistical test (e.g. population stability index, KS test) — alerting if incoming applicant data drifts meaningfully from what the model was trained on.

**40. How would you monitor for concept drift (the *relationship* between features and churn changing over time, not just the inputs)?**
Once ground-truth churn outcomes become available for scored customers (e.g. 90 days later), periodically recompute precision/recall/F1 on that cohort and compare against the original offline evaluation — a widening gap signals the model's learned patterns no longer match current customer behavior (e.g. after a pricing change or new competitor).

**41. What logging would you add around each prediction call for production observability?**
Structured logs per request: timestamp, a request ID, the (hashed/anonymized if needed) input features, model version used, output probability and class, and latency — enabling later debugging of specific predictions, drift analysis, and A/B comparison between model versions in production.

**42. If this app needed to serve predictions via an API (not just an interactive Streamlit UI), what would you change architecturally?**
Separate the model-serving logic from the UI entirely: wrap `load_model`/`preprocessor_input`/`predict` in a lightweight API service (e.g. FastAPI) that loads artifacts once at startup, exposes a `/predict` endpoint with request/response schema validation (e.g. Pydantic), and let Streamlit (or any other client) call that API rather than embedding model logic directly in the UI layer.

**43. Why might `st.cache_resource`-based model loading be problematic if this app were horizontally scaled across multiple instances behind a load balancer?**
Each instance independently loads its own copy of the model/scaler/data into memory at startup — fine for a small model, but for large models or high instance counts this duplicates memory usage and cold-start latency across every replica, versus a shared model-serving backend that all app instances call over the network.

**44. How would you add automated tests for the `preprocessor_input` function specifically?**
Unit tests with fixed known inputs asserting: (1) output shape matches `len(template_columns)`, (2) a known categorical value produces the expected one-hot pattern, (3) `tenure` binning edges are handled correctly (e.g. tenure=0 and tenure=72 boundary cases), and (4) an unseen/malformed category doesn't crash but is handled per the intended fallback behavior.

**45. What would a CI/CD pipeline for this project run before allowing a deploy?**
Linting, the preprocessing consistency tests (Q9/Q44), a smoke test that loads the actual production model artifact and runs one known input through the full app pipeline checking the output is in `[0, 1]`, and a check that all `requirements.txt` versions match what's pinned/expected — failing the build if any of these don't pass.

**46. If a customer disputes a churn prediction that led to them being denied a retention offer, what would you need logged to investigate it?**
The exact input feature vector submitted, model version/artifact hash used at that time, the raw probability output and threshold applied, and ideally the SHAP/feature-importance breakdown for that specific prediction — none of which is currently captured by this app beyond the transient UI session.

---

## 📊 Section 6: Business Alignment & Explainability (Q47–Q50)

**47. The EDA found "Electronic check" payment method has a dramatically higher churn rate. How should this inform a production feature rather than just being an EDA footnote?**
It's a strong signal worth actively monitoring as a leading indicator — the retention/marketing team could get an automated alert when a cohort's electronic-check share grows, independent of the ML model, since this is a simple, interpretable business rule that complements (not replaces) the model.

**48. Given the deployed AdaBoost model isn't inherently interpretable per-prediction, how would you explain an individual churn prediction to a non-technical retention agent?**
Use SHAP values (works with tree-based/ensemble models like AdaBoost with decision tree base estimators) to generate a per-customer breakdown of which features pushed the churn probability up or down, then surface the top 2-3 factors in plain language in the UI (e.g. "Short tenure and month-to-month contract are the biggest drivers here") rather than just the raw probability.

**49. The project shows Month-to-month contracts and Fiber optic internet correlate with high churn. Is it appropriate for the model to use these as direct levers for "fixing" churn, or is this a case of correlation vs. causation?**
This is a classic correlation-vs-causation trap — Month-to-month customers may churn more because they're inherently less committed (self-selection), not *because* of the contract type itself; simply forcing them onto longer contracts might not reduce true churn risk and could backfire (frustration, cancellations). Any intervention based on this insight should ideally be validated with a controlled experiment (e.g. an A/B test offering contract incentives) before being treated as causal.

**50. If you had to pitch one concrete next step to take this from "notebook + Streamlit demo" to a genuinely production-grade system, what would it be?**
Fix the scaler/preprocessing mismatch first (Q1) since it's silently corrupting every live prediction — that's a correctness bug, not a nice-to-have. Then consolidate model artifacts into one versioned, governed bundle (preprocessing + model + metadata) loaded together, add `random_state` to all splits for reproducibility, and stand up basic drift monitoring before treating this as anything more than a prototype.

---

*Companion reference for the Telecom Customer Churn Prediction project (imbalanced classification, multi-model comparison, Streamlit deployment).*
