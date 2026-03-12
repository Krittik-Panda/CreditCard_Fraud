# Baseline dataset used in examples

Assume a dataset of **10,000 transactions** where **100 are fraudulent** (1% fraud).
Majority (legit) = 9,900; Minority (fraud) = 100.

---

# 1) Random oversampling

**What it is / how it works**
Duplicate randomly picked minority class samples until you reach the target balance. Implementation: sample with replacement from fraud rows.

**Numeric example**
To 1:1 balance → generate 9,800 duplicates of fraud → new minority = 9,900 → total = 19,800.

**When to use**

* Quick baseline when sample size of minority is very small and you want a simple fix.
* When features are complicated and you don’t trust synthetic interpolation.

**Assumptions**

* Duplicating examples will help the classifier learn minority regions without introducing false patterns.

**Risks & failure modes**

* **Overfitting:** model can memorize duplicated minority records (esp. tree-based models). Performance on validation/test will look optimistic if you leak duplicates into validation.
* **No new information:** duplicates do not expand feature-space coverage.
* **Temporal/data leakage:** if you duplicate across time-ordered folds you’ll leak signal.

**Tradeoffs**

* Fast and deterministic; but increases dataset size and overfitting risk.
* Works only if you always oversample *only* in the training fold (never before splitting).

**Practical tips**

* Use only inside training folds.
* Combine with strong regularization or ensemble methods to reduce memorization.

---

# 2) Random undersampling

**What it is / how it works**
Randomly remove majority-class (legit) samples until balance (or desired ratio) is reached.

**Numeric example**
To 1:1 balance → randomly drop 9,800 legit → remaining majority = 100 → total = 200.

**When to use**

* When majority class is extremely large and redundant.
* When training time / memory is the bottleneck.

**Assumptions**

* Majority samples are redundant; removing many won’t remove important decision-boundary information.

**Risks & failure modes**

* **Information loss:** you may remove majority examples near the decision boundary → worse generalization.
* **High variance:** results depend heavily on which majority samples you keep.
* **Class boundary shift:** if undersampling removes legitimate variations the model will mislearn the legit distribution.

**Tradeoffs**

* Reduces compute and class imbalance but risks removing informative examples.
* Works better if you combine with smarter undersampling (cluster-based) or repeat experiments with different random seeds.

**Practical tips**

* Prefer to undersample to a *less extreme* ratio (e.g., 1:5) instead of 1:1 if data is scarce.
* Run repeated experiments with different seeds and ensemble the results.

---

# 3) Tomek Links (undersampling via cleaning)

**What it is / how it works**
A **pairwise cleaning** method. For a pair of samples (x, y) from different classes, if each is the other’s nearest neighbor (mutual nearest neighbors), that pair is a *Tomek link*. The usual action: remove the majority-class sample from each Tomek link. This cleans overlapping border samples.

**Numeric example**
If 200 majority samples form Tomek pairs with minority samples, removing them reduces majority from 9,900 → 9,700. (You usually don’t remove minority here.)

**When to use**

* To **clean class overlap** after oversampling (commonly used after SMOTE).
* When you want to remove noisy majority points that are ambiguous.

**Assumptions**

* Mutual nearest neighbors indicate noisy/overlap points; removing the majority helps separate classes.

**Risks & failure modes**

* **Edge-case removal:** if fraud and legit legitimately are close in feature-space (real overlap), removing majority can distort real distribution.
* **Ineffective in high dimensions**: nearest neighbor relations become noisy in high-D spaces.
* **Label noise sensitivity:** if labels are noisy, Tomek can remove valid examples.

**Tradeoffs**

* Small, conservative reduction of majority; safer than aggressive undersampling.
* Works well after SMOTE (SMOTE+Tomek is common).

**Practical tips**

* Always apply after oversampling (if used).
* Use distance metric appropriate to your features (e.g., scaled numeric features).

---

# 4) Cluster centroids (undersampling)

**What it is / how it works**
Run clustering (e.g., k-means) on the majority class and replace groups of majority samples with their cluster centroids (one centroid per cluster). This is a **representative undersampling**: fewer, more “typical” majority points kept.

**Numeric example**
If you request 100 centroids for the majority class → majority becomes 100 (centroids) → total samples = 200.

**When to use**

* When majority class contains many redundant points and decision regions are well-clustered.
* When you need dramatic dataset size reduction but want to preserve representative structure.

**Assumptions**

* Majority class clusters are meaningful and centroids represent the local distribution for that cluster.

**Risks & failure modes**

* **Averaging artifacts:** centroids are synthetic averages — they may fall into minority regions if clusters overlap.
* **Loss of boundary examples:** centroids often sit in cluster centers, not on the class boundary where informative samples may lie.
* **K-means assumptions:** spherical clusters; poor performance if clusters are irregular.

**Tradeoffs**

* Better than random undersampling at preserving global distribution, but can wash out boundary detail needed to separate classes.

**Practical tips**

* Choose number of centroids carefully (cross-validate).
* Consider using cluster centroids to produce a smaller training set and then fine-tune on borderline real examples.

---

# 5) SMOTE (Synthetic Minority Oversampling Technique)

**What it is / how it works**
SMOTE generates new synthetic minority samples by **interpolating** between a minority sample and one of its k nearest minority neighbors:

new = x_i + λ * (x_nn − x_i), where λ ∈ (0,1).

**Numeric example**
To 1:1 balance → generate 9,800 synthetic fraud examples via interpolation → minority = 9,900 → total = 19,800.

**When to use**

* When you want to expand minority coverage without duplicating exact samples.
* Works well when minority class has enough examples to define local neighborhoods.

**Assumptions**

* Minority class lies on continuous manifolds where interpolation between neighbors produces realistic samples.
* Features are numeric or can be handled by a variant (SMOTENC for mixed categorical).

**Risks & failure modes**

* **Synthetic samples cross class boundary:** if minority is adjacent to majority, SMOTE may create points inside majority region → increased false positives.
* **Exacerbates class overlap:** unless followed by cleaning (Tomek/ENN).
* **Not for tiny minority:** if minority has very few examples (<5–10), kNN is unreliable → poor synthetic points.
* **Categorical variables:** standard SMOTE cannot handle pure categorical features correctly (use SMOTENC or discrete methods).

**Tradeoffs**

* Better than duplication in expanding the decision region, but can create unrealistic samples if distribution is complex.
* Use with cleaning methods (Tomek or Edited Nearest Neighbors) to remove harmful synthetic points.

**Practical tips**

* Apply **after** splitting into train/validation (only on training fold).
* Standardize/scale numeric features before SMOTE.
* For categorical features use **SMOTENC** or variant that respects categorical distances.
* Use SMOTE **limited** to not create a huge training set: e.g., move to 1:10 or 1:4, not necessarily 1:1.

---

# 6) SMOTE + Tomek Links (combined)

**What it is / how it works**
First oversample minority with SMOTE, then use Tomek Links to remove majority samples (and sometimes synthetic minority) that form Tomek pairs. The idea: create synthetic minority coverage, then clean overlapping / noisy majority examples.

**Numeric example workflow**

1. SMOTE → minority 9,900 (total 19,800).
2. Tomek removal → drop, say, 300 majority samples that are Tomek pairs → final majority = 9,600; minority unchanged or slightly reduced if minority Tomeks removed → final total ≈ 19,500.

**When to use**

* When SMOTE alone creates overlap—Tomek helps clean the decision boundary.
* A common, effective pipeline for many imbalanced problems.

**Assumptions**

* Tomek links correctly identify noisy overlaps after synthetic generation.

**Risks & failure modes**

* If many real minority instances are near majority, Tomek can remove too many majority points or even minority samples (if configured), harming model.
* If features are high-dimensional/noisy, nearest-neighbor relations used by both SMOTE and Tomek are unreliable.

**Tradeoffs**

* Usually improves over SMOTE alone but still requires careful metric selection and validation.

**Practical tips**

* Use SMOTE + Tomek as a default if you must oversample; still validate with temporal hold-out and precision/recall metrics.
* Tune k for SMOTE and check how many Tomek links are removed.

---

# Cross-cutting, project-critical issues for credit-card fraud

These are the common traps that break any of the above methods in a fraud pipeline:

1. **Time dependency (temporal leakage)**
   Fraud tasks are temporal. Never randomly resample across time boundaries. Always split chronologically: train on older transactions, validate/test on newer transactions. Otherwise oversampling will leak future patterns.

2. **Evaluation metric choice**
   Don’t optimize for accuracy. Use **precision, recall, F1**, and especially **Precision–Recall AUC (AUC-PR)** and cost-sensitive metrics (cost matrix). For fraud, low false-positive rate is important operationally — but missing fraud costs money; choose metric that matches business cost.

3. **Perform resampling only inside training folds**
   If using cross-validation, apply SMOTE/undersampling inside each training fold. Never fit SMOTE on the whole dataset.

4. **Feature scaling and transformations**
   SMOTE and Tomek use distances — **scale numeric features** before applying. For categorical features use SMOTENC or encode carefully (target encoding can leak).

5. **Handling categorical features**
   SMOTE works on continuous features. For mixed data use **SMOTENC** or specialized techniques (CTGAN, SMOTE variants for categorical).

6. **Label noise and attack vectors**
   Fraud datasets often have mislabeled fraud/legit. Synthetic generation or cleaning can amplify label-noise effects. Inspect borderline cases manually.

7. **High-dimensionality**
   k-NN and nearest-neighbor based methods degrade in high dimensions — distances become meaningless → synthetic samples or Tomek decisions are noisy.

8. **Class imbalance severity & minority sample count**
   If you have fewer than ~50–100 minority examples, synthetic methods are fragile. Consider anomaly detection or one-class models instead.

9. **Deploy-time drift**
   Fraud patterns drift. Resampling that made sense on historical distribution can be harmful later. Keep models adaptable and monitor.

---

# Minimal, concrete pipeline I would run — the *smallest* change set that actually works

(Apply these in order; each is necessary)

1. **Split chronologically** into train / validation / test (no leakage).
2. **Feature engineering + scaling**: prepare numeric features and encode categoricals (no target leakage). Standardize numeric features before any distance-based step.
3. **On training fold only**: try two competing pipelines:

   * A. *SMOTE (to moderate ratio e.g., 1:5) → Tomek links*
   * B. *Cluster centroids (reduce majority moderately) + light random undersampling*
     Evaluate both.
4. **Modeling**: try a robust classifier (LightGBM/XGBoost with balanced objective or class weights) — compare with cost-sensitive learning (class_weight or custom loss).
5. **Evaluation**: use time-based validation; measure AUC-PR, recall@precision thresholds, and business cost metrics.
6. **If minority < 50**: avoid SMOTE; favor anomaly detection or feature enrichment.
7. **Monitor drift**: set monitoring for performance decay and automatic re-training triggers.

These are the smallest changes which prevent the most common failures (leakage, overfitting, bad metrics).

---

# Hard questions you must answer about your dataset (answer these before picking a resampling method)

1. How many fraud examples do you have total and per fraud subtype? (If subtypes exist, SMOTE can collapse subtypes.)
2. Is the dataset **time-ordered**? If yes, can you enforce chronological splits?
3. Are features mostly numeric, categorical, or mixed? (SMOTE needs numeric or SMOTENC.)
4. Do you have label noise or weak labels? Can you verify a sample of borderline cases?
5. What is the **business cost** of false positives vs false negatives (monetary or operational)?
6. How often does the fraud pattern change (drift)? Will you retrain frequently?

If you can’t answer these now, don’t blindly apply SMOTE or aggressive undersampling.

---

# Final prioritized checklist (do this immediately)

1. Split data chronologically.
2. Choose evaluation metric tied to business cost (AUC-PR + recall@fixed-precision).
3. Scale numerics; encode categoricals using safe encoders.
4. Run two experiments: (SMOTE→Tomek) and (class-weight or focal loss + no resampling). Compare on time-split validation.
5. If SMOTE chosen: use SMOTENC for categoricals, limit oversampling ratio (don’t force 1:1 unless justified), then apply Tomek.
6. If undersampling chosen: prefer cluster centroids or repeated undersampling + ensemble to reduce variance.
7. Validate on an unseen chronological test set and monitor for drift.

---

