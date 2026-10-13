# ML Assignment — Master Plan (v2)

_Fundamentals of Machine Learning — Programming Assignment · group of 2 · 40 marks._
_This version replaces the earlier plan (which was built around the Credit Card dataset).
It reflects the switch to the Gas Sensor dataset, the mandatory PCA + feature-engineering
instructions, and alignment to the lab sheets and lectures._

---

## 0. Scope — confirmed
The brief says apply an algorithm "under supervised, unsupervised **or** reinforcement
learning", and the rubric has **two separate marking tables** (Supervised/Unsupervised,
and Reinforcement). → You choose **one paradigm**. We are doing **supervised learning**
with **two distinct algorithms** (team of 2). No unsupervised/RL components are required.
_(A one-line email to the lecturer remains cheap insurance, but the document is clear.)_

---

## 1. The project
**Title:** Gas Classification under Sensor Drift — a comparative study of Logistic
Regression and Random Forest with feature selection and PCA.

**Problem:** classify which of **6 gases** a 16-sensor chemical array is detecting from
its **128 response features** (supervised multiclass classification), then study how
feature engineering, feature selection, PCA, and sensor drift over time affect accuracy.

---

## 2. Dataset
- **Name:** Gas Sensor Array Drift at Different Concentrations
- **Source:** UCI ML Repository, **ID 270** — <https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+at+different+concentrations>
- **Citation:** Vergara, A. et al. (2012). *Gas Sensor Array Drift Dataset at Different Concentrations.* UCI ML Repository. https://doi.org/10.24432/C5MK6M (CC BY 4.0)
- **Size:** 13,910 measurements × 128 continuous features
- **Target:** 6 gases — Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene (mild imbalance 1,641–3,009)
- **Structure:** 16 sensors × 8 features each; 10 acquisition **batches** over 36 months (basis of the drift study)
- **Quality:** no missing values; features span ~6 orders of magnitude → standardisation essential
- _Loading:_ notebook cites UCI 270 and loads a byte-identical public CSV mirror for one-line reproducibility.

### Why we chose this dataset (rubric criterion 1)
- **Not a tutorial dataset.** It is a genuine research dataset (UC San Diego), unlike Iris/Titanic/Wine, directly satisfying the "should not be part of a tutorial" requirement.
- **Complex enough.** 13,910 samples × **128 continuous features** and 6 classes give real analytical depth, well beyond a toy problem.
- **PCA has genuine value.** With 128 correlated features, dimensionality reduction is a real investigation, not a box-tick — this was mandatory (sir) and maps onto the rubric's "dimensionality reduction".
- **Supports a strong metric story.** Multiclass with mild imbalance justifies macro-F1 / balanced accuracy beyond plain accuracy — the "appropriate metrics" the brief asks for.
- **Enables a standout experiment.** The 10 acquisition batches over 36 months let us study **sensor drift** (train early, test later) — a real-world generalisation question most submissions won't have, powering the critical-analysis criterion.
- **Clean but realistic.** No missing values, yet wildly different feature scales → a clear, defensible preprocessing/standardisation rationale.
- **Publicly hosted and citable** (UCI, CC BY 4.0) — satisfies the "hosted somewhere" and citation requirements.

---

## 3. Algorithms (one per member)
- **Member A → Logistic Regression** (linear, interpretable baseline) — Lab 05
- **Member B → Random Forest** (non-linear ensemble) — Lab 06
- **Justification:** a six-model screen (LR, KNN, Decision Tree, RF, Naive Bayes, SVM —
  Labs 05–09) shows *why* these two are the chosen distinct pair, without adding required
  deliverables. Naive Bayes fails (correlated features violate its independence assumption);
  KNN scores high on a random split but is non-interpretable and drift-fragile.

---

## 4. Method, aligned to the lab sheets
**Preprocessing (Lab 02 — Data Preprocessing, Feature Engineering & Feature Selection)**
- Data audit (dtypes, missing %, duplicates, class & batch balance, scale range)
- Skew handling via signed-log transform; StandardScaler (MinMax vs Standard discussed), fit on train only (no leakage)
- **Feature engineering:** per-sensor aggregates (mean/range), **cross-sensor ratios** (drift-robust), signed-log features
- **Feature selection:** correlation redundancy + **ANOVA F-test** (`f_classif`) + **Mutual Information** (`mutual_info_classif`); Chi-square skipped (no categoricals) — with the Lab-02 caveat that *statistical significance ≠ predictive usefulness*

**Dimensionality reduction (Lab 11 — PCA)**
- Standardise → PCA; cumulative explained-variance curve; components for 95% variance (~13)
- 2-component scatter; component sweep `[2,5,8,10,15,20,25,30]` with 5-fold stratified CV
- "How many components" analysis: accuracy vs dimensionality

**Modelling & evaluation (Labs 05/06/08)**
- Pipelines with StandardScaler; `class_weight` for imbalance
- Compare LR & RF across representations: all-128 / engineered / selected / PCA
- Metrics: accuracy, macro precision/recall/F1, balanced accuracy, one-vs-rest ROC-AUC, confusion matrices, `classification_report`
- 5-fold stratified cross-validation
- **GridSearchCV** hyper-parameter tuning (Lab 08) — the "how we obtain higher accuracy" story
- Interpretability: RF feature importances + LR coefficients (two lenses)

**Key experiment — sensor drift**
- Train on early batches, test on later ones (multiple configs for robustness)
- Per-batch degradation curve; train-vs-test gap as evidence of overfitting

---

## 5. Results so far (real, from the data)
| Setting | Logistic Regression | Random Forest |
|---|---:|---:|
| All 128 features (test acc) | 0.990 | 0.995 |
| Correlation-filtered (58 feats) | 0.984 | 0.995 |
| PCA-95% (13 components) | 0.926 | 0.992 |
| CV macro-F1 (PCA-13) | 0.919 ± 0.005 | 0.990 ± 0.002 |
| **Drift (train 1–5, test 6–10)** | **0.764** | 0.620 |

**Headline findings**
- PCA compresses 128 → ~13 components (95% variance) with RF holding ~0.99 accuracy.
- On a random split RF wins; **under sensor drift the ranking reverses** — the simpler
  Logistic Regression generalises better, because RF overfits the training batches'
  sensor characteristics. This is the top-marks critical-analysis centrepiece.

---

## 6. Deliverables
**Mandatory (any missing = zero marks)**
- `members.txt` — both members' IDs + emails
- `submission.txt` — dataset link (UCI 270) + GitHub repo link + YouTube link
- GitHub repo with **detailed commit history from inception**
- YouTube video (each member ≤5 min)
- **PDF report** (main marked item) — problem, dataset, methodology, results, discussion (limitations/future work), comparing algorithms' accuracy with appropriate metrics
- **All source code as text in the report appendix** (not screenshots)
- Bundle zipped as `ML-assignment.zip`; one member uploads
- Turnitin similarity < 20%

**Recommended**
- Proper repo README (replace the current one-line file; update dataset link to UCI 270)
- Viva study guide

**Only the team can do:** the video, the real commit history, the viva, and names/IDs/emails + each member's individual-contribution statement.

---

## 7. Status & next steps
**Done:** dataset verified; full analysis + executed notebook (`Gas_Sensor_Drift_ML.ipynb`) with 8 figures.

**In progress — notebook upgrades:**
1. Feature engineering + feature selection (Lab-02 aligned)
2. Robust drift protocol (multiple train/test configs)
3. Train-vs-test gap (evidence RF overfits)
4. GridSearchCV hyper-parameter tuning
5. LR coefficient interpretability
6. Load from official UCI source (with mirror fallback)

**Then:** PDF report → `members.txt`/`submission.txt`/README → viva guide → zip.

**Flags to fix:** the old `README.md` and `ML_Assignment_Plan.md` still reference the
previous dataset (UCI 350) — update both to Gas Sensor (UCI 270).

---

## 8. Resources on hand
- **Assignment brief** + **marking rubric** (read in full)
- **Lectures:** 1 (Intro), 2 (Preprocessing/FE/FS), 3 (Linear Regression), 4 (LogReg/kNN), 5 (Decision Trees/RF), 6 (SVM), 7 (tbc), 9 (Unsupervised), Bayesian, RL
- **Lab sheets:** Lab 02 (preprocessing/FE/FS), 04, 05 (LogReg/KNN), 06 (DT/RF/ensembles), 07 (SVM), 08 (evaluation/imbalance/GridSearch), Bayesian, 10 (clustering), 11 (PCA)
- Most relevant and already mirrored in the notebook: **Lab 02** (preprocessing) and **Lab 11** (PCA); evaluation conventions from **Lab 08**.
