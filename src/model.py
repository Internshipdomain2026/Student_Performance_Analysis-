"""
Student Performance Analysis and Prediction System
Run from repository root: python src/model.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance_dataset.csv"
OUT = ROOT / "outputs"
VIZ = OUT / "visualizations"
EVAL = OUT / "evaluation_results"
for folder in [VIZ, EVAL]:
    folder.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape before cleaning:", df.shape)
    print("Missing values by column:\n", df.isna().sum())
    print("Duplicate student IDs:", df.duplicated(subset=["student_id"]).sum())
    df = df.drop_duplicates(subset=["student_id"]).copy()

    features = ["study_hours_per_day", "attendance_rate", "previous_exam_marks",
                "assignment_score", "class_participation", "absences"]
    target = "final_exam_marks"
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([("numeric", numeric_pipeline, features)])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    # Optional classifier; threshold of 40/100 is only a demonstration assumption.
    y_pass = (y >= 40).astype(int)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_pass, test_size=0.20, random_state=42, stratify=y_pass
    )
    classifier = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    classifier.fit(X_train_c, y_train_c)
    class_predictions = classifier.predict(X_test_c)
    metrics = {
        "rows": int(len(df)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
        "CV_R2_mean": float(cv_scores.mean()), "CV_R2_std": float(cv_scores.std()),
        "classification_accuracy": float(accuracy_score(y_test_c, class_predictions)),
        "classification_precision": float(precision_score(y_test_c, class_predictions, zero_division=0)),
        "classification_recall": float(recall_score(y_test_c, class_predictions, zero_division=0)),
        "classification_f1": float(f1_score(y_test_c, class_predictions, zero_division=0))
    }
    (EVAL / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"actual_marks": y_test.to_numpy(), "predicted_marks": predictions,
                  "absolute_error": np.abs(y_test.to_numpy()-predictions)}).to_csv(EVAL/"predictions.csv", index=False)

    plt.figure(figsize=(8,5)); plt.hist(df[target], bins=20, edgecolor="black")
    plt.title("Distribution of Final Examination Marks"); plt.xlabel("Final marks"); plt.ylabel("Students")
    plt.tight_layout(); plt.savefig(VIZ/"final_marks_histogram.png",dpi=160); plt.close()
    plt.figure(figsize=(8,5)); plt.scatter(df.study_hours_per_day,df[target],alpha=.6)
    plt.title("Study Hours vs Final Marks"); plt.xlabel("Study hours/day"); plt.ylabel("Final marks")
    plt.tight_layout(); plt.savefig(VIZ/"study_hours_scatter.png",dpi=160); plt.close()
    plt.figure(figsize=(8,5)); plt.scatter(df.attendance_rate,df[target],alpha=.6)
    plt.title("Attendance vs Final Marks"); plt.xlabel("Attendance (%)"); plt.ylabel("Final marks")
    plt.tight_layout(); plt.savefig(VIZ/"attendance_scatter.png",dpi=160); plt.close()
    df["study_hour_group"] = pd.cut(df.study_hours_per_day,bins=[0,2,4,6,9],labels=["0-2","2-4","4-6","6-9"])
    plt.figure(figsize=(8,5)); df.boxplot(column=target,by="study_hour_group")
    plt.title("Final Marks by Study-Hour Group"); plt.suptitle("")
    plt.xlabel("Study hours/day"); plt.ylabel("Final marks")
    plt.tight_layout(); plt.savefig(VIZ/"study_group_boxplot.png",dpi=160); plt.close()
    corr=df[features+[target]].corr(numeric_only=True)
    plt.figure(figsize=(8,6)); plt.imshow(corr,aspect="auto",vmin=-1,vmax=1)
    plt.xticks(range(len(corr.columns)),corr.columns,rotation=45,ha="right")
    plt.yticks(range(len(corr.columns)),corr.columns); plt.colorbar(label="Correlation")
    plt.title("Correlation Heatmap"); plt.tight_layout()
    plt.savefig(VIZ/"correlation_heatmap.png",dpi=160); plt.close()
    plt.figure(figsize=(7,6)); plt.scatter(y_test,predictions,alpha=.7); plt.plot([0,100],[0,100],"--")
    plt.title("Actual vs Predicted Marks"); plt.xlabel("Actual marks"); plt.ylabel("Predicted marks")
    plt.tight_layout(); plt.savefig(VIZ/"actual_vs_predicted.png",dpi=160); plt.close()
    print("\nREGRESSION EVALUATION")
    for key in ["MAE","RMSE","R2","CV_R2_mean","CV_R2_std"]: print(f"{key}: {metrics[key]:.4f}")
    print("\nOPTIONAL CLASSIFICATION")
    for key in ["classification_accuracy","classification_precision","classification_recall","classification_f1"]:
        print(f"{key}: {metrics[key]:.4f}")
    print("Outputs saved under outputs/")

if __name__ == "__main__":
    main()
