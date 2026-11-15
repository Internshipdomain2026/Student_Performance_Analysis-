# Student Performance Analysis and Prediction System

## Project overview
A Python machine-learning project that explores academic performance patterns and predicts final examination marks from study time, attendance, previous examination marks, assignment scores, participation and absences.

## Dataset source and attribution
The bundled `data/student_performance_dataset.csv` is a **synthetic, reproducible demonstration dataset** (600 records) generated with NumPy using a fixed random seed. It is not real student data and must not be described as actual student records. It is included so the project runs immediately.

A public real-world reference dataset is available from the UCI Machine Learning Repository:
- https://archive.ics.uci.edu/dataset/320/student+performance
- UCI describes 649 student records from two Portuguese schools. Its fields include `studytime`, `absences`, `G1`, `G2` and `G3`. It does not contain direct attendance percentage or daily study hours; `studytime` is an ordinal weekly study-time category.

## Technologies
Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn and Jupyter Notebook.

## Folder structure
```text
Student_Performance_Analysis/
├── data/student_performance_dataset.csv
├── notebooks/student_performance_analysis.ipynb
├── src/model.py
├── outputs/visualizations/
├── outputs/evaluation_results/
├── screenshots/
├── Project_Report.pdf
├── README.md
└── requirements.txt
```

## Installation
Python 3.9+ recommended:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Python file
From the repository root:
```bash
python src/model.py
```

## Run the notebook
```bash
jupyter notebook notebooks/student_performance_analysis.ipynb
```
Run all cells from top to bottom.

## Methodology
1. Load and inspect the CSV.
2. Check missing values and duplicate IDs; remove duplicate IDs.
3. Split into 80% training and 20% testing records.
4. Fit median imputation and standardization only on training data using a Pipeline.
5. Train Linear Regression to predict final examination marks.
6. Evaluate with MAE, RMSE and R²; run 5-fold cross-validation on training data.
7. Optional Logistic Regression classifies marks >= 40 as pass (illustrative threshold).
8. Save visualizations, metrics and predictions under `outputs/`.

## Limitations
Results are illustrative because the bundled dataset is synthetic. Predictions are estimates, not decisions about student ability. A real deployment requires authorized, representative student data, institution-specific grading rules, privacy protection and fairness evaluation.
