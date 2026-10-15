# Gas Classification under Sensor Drift

A comparative machine-learning study that classifies **6 gases** from a 16-sensor
chemical array (128 response features), using **Logistic Regression** and
**Random Forest**, with feature engineering, feature selection and **PCA**
dimensionality reduction. It includes a **sensor-drift experiment** that measures
how the models generalise to data collected in later time periods.

_Fundamentals of Machine Learning (IT5022) - Programming Assignment._

---

## Group members
| Student ID | Name | Track |
|---|---|---|
| MS26923024 | Saara M.K.F | Random Forest; PCA; six-algorithm screening; sensor-drift analysis |
| MS26924250 | Dhanushikan V | Logistic Regression; data audit; EDA; standardisation; feature engineering & selection |

## Problem
Supervised multiclass classification: given 128 sensor-response features, predict
which of six gases (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene) is
present - and test how reliably the models perform on later acquisition batches.

## Dataset
- **Gas Sensor Array Drift at Different Concentrations** - UCI ML Repository, ID 270
- 13,910 measurements x 128 features, 6 classes, 10 acquisition batches over ~36 months
- Link: <https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+at+different+concentrations>
- Citation: Vergara, A. et al. (2012). *Gas Sensor Array Drift Dataset at Different Concentrations.* UCI ML Repository. https://doi.org/10.24432/C5MK6M (CC BY 4.0)

## Methods
- **Preprocessing:** data audit, standardisation (fit on training data only; re-fit inside each CV fold)
- **Feature engineering:** signed-log transforms, per-sensor means/ranges, cross-sensor ratios (176 features)
- **Feature selection:** ANOVA F-test and Mutual Information, with `SelectKBest` inside the CV pipeline
- **Dimensionality reduction:** PCA (95% variance -> 13 components) + a component sweep
- **Algorithms:** Logistic Regression and Random Forest, chosen via a six-model screen
  (LR, KNN, Decision Tree, RF, Naive Bayes, SVM) on original vs PCA features
- **Evaluation:** accuracy, macro-F1, balanced accuracy, one-vs-rest ROC-AUC, confusion
  matrices, stratified cross-validation, and training-only `GridSearchCV` tuning
- **Key experiment:** train on early batches, test on later batches (sensor drift)

## Key results
- **Random split (original 128 features):** Random Forest ~0.993 accuracy, Logistic Regression ~0.991.
- **PCA (95% -> 13 components):** Random Forest retains ~0.991; Logistic Regression drops to ~0.936
  (the linear model is more affected by PCA than the ensemble).
- **Sensor drift (train early batches -> test later):** accuracy falls to ~0.56-0.63; the ranking is
  **period-dependent** (LR higher in two of three chronological splits, RF in one).
- **Takeaway:** a random split substantially overstates real-world performance under sensor drift.

## Repository structure
```
.
├── FML_Assignment.ipynb     # main analysis notebook (run top-to-bottom)
├── ML-assignment.pdf        # final report
├── members.txt              # group members (IDs + emails)
├── submission.txt           # dataset / repo / video links
├── data/                    # (optional) batch1.dat ... batch10.dat for offline runs
└── README.md
```

## How to run
```bash
python -m venv ml-assignment
# Windows: ml-assignment\Scripts\activate    |    macOS/Linux: source ml-assignment/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
jupyter notebook FML_Assignment.ipynb   # or open in VS Code and Run All
```
Run the cells in order. A fixed `random_state = 42` makes results reproducible.
The notebook downloads the dataset on first run (needs internet); alternatively place
the ten `batch*.dat` files in a `data/` folder to run offline.

## Notes
- Classical (non-deep-learning) machine learning only, per the assignment brief.
- Batch and concentration are kept as metadata and are never used as model inputs.
