# Gas Classification under Sensor Drift

A comparative machine-learning study classifying **6 gases** from a 16-sensor
chemical array, using **Logistic Regression** and **Random Forest**, with feature
engineering, feature selection, and **PCA** dimensionality reduction. Includes a
**sensor-drift experiment** measuring how the models degrade over 36 months.

_Fundamentals of Machine Learning — Programming Assignment (group of 2)._

---

## Problem
Supervised multiclass classification: given 128 sensor-response features, predict
which of 6 gases (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene) is present.

## Dataset
- **Gas Sensor Array Drift at Different Concentrations** — UCI ML Repository, ID 270
- 13,910 measurements × 128 features, 6 classes, 10 acquisition batches over 36 months
- Link: https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+at+different+concentrations
- Citation: Vergara, A. et al. (2012). *Gas Sensor Array Drift Dataset at Different Concentrations.* UCI ML Repository. https://doi.org/10.24432/C5MK6M (CC BY 4.0)

## Methods
- **Preprocessing:** data audit, signed-log transform, standardisation (fit on train only)
- **Feature engineering:** per-sensor aggregates and drift-robust cross-sensor ratios
- **Feature selection:** correlation redundancy, ANOVA F-test, Mutual Information
- **Dimensionality reduction:** PCA (95% variance) + component sweep with cross-validation
- **Algorithms:** Logistic Regression (Member A) and Random Forest (Member B); a six-model
  screen (LR, KNN, Decision Tree, RF, Naive Bayes, SVM) justifies the choice
- **Evaluation:** accuracy, macro precision/recall/F1, balanced accuracy, ROC-AUC,
  confusion matrices, 5-fold stratified cross-validation, GridSearchCV tuning
- **Key experiment:** train on early batches, test on later batches (sensor drift)

## Key results
- PCA compresses 128 features to ~13 components (95% variance) with Random Forest
  retaining ~0.99 accuracy on a random split.
- Under sensor drift, the ranking reverses: Logistic Regression (~0.76) generalises
  better than Random Forest (~0.62), which overfits the training batches.

## Repository structure
```
.
├── Gas_Sensor_Drift_ML.ipynb   # main analysis notebook
├── report.pdf                  # final report (main deliverable)
├── members.txt                 # group members
├── submission.txt              # dataset / repo / video links
└── README.md                   # this file
```

## How to run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo
jupyter notebook Gas_Sensor_Drift_ML.ipynb
```
Run the cells in order. A fixed `random_state=42` makes results reproducible.
