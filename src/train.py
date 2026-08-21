import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Ty le tham chieu cua lop duong (thu nhap > 50K) trong bo du lieu Adult goc.
REFERENCE_POSITIVE_RATIO = 0.248
POSITIVE_RATIO_TOLERANCE = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: canh bao lech lac phan phoi du lieu (data drift)
    positive_ratio = float(y_train.mean())
    ratio_diff = abs(positive_ratio - REFERENCE_POSITIVE_RATIO)
    if ratio_diff > POSITIVE_RATIO_TOLERANCE:
        print(
            f"CANH BAO: ty le lop duong trong du lieu huan luyen la {positive_ratio:.4f}, "
            f"lech {ratio_diff:.4f} so voi ty le tham chieu {REFERENCE_POSITIVE_RATIO:.4f} "
            f"(nguong cho phep {POSITIVE_RATIO_TOLERANCE:.2f})."
        )
    else:
        print(f"Ty le lop duong trong du lieu huan luyen: {positive_ratio:.4f} (binh thuong).")

    with mlflow.start_run():

        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)

        # Bonus 2: quet nguong quyet dinh tu 0.1 den 0.9 (buoc 0.05) de tim f1 toi uu
        proba = model.predict_proba(X_eval)[:, 1]
        best_threshold = 0.5
        best_threshold_f1 = f1
        for threshold in [round(0.1 + 0.05 * i, 2) for i in range(17)]:
            preds_at_threshold = (proba >= threshold).astype(int)
            f1_at_threshold = f1_score(y_eval, preds_at_threshold)
            if f1_at_threshold > best_threshold_f1:
                best_threshold_f1 = f1_at_threshold
                best_threshold = threshold
        print(
            f"Nguong mac dinh 0.5: f1={f1:.4f} | "
            f"Nguong toi uu {best_threshold}: f1={best_threshold_f1:.4f}"
        )

        # Bonus 3: bao cao precision/recall va confusion matrix chi tiet
        cm = confusion_matrix(y_eval, preds)
        precision_per_class = precision_score(y_eval, preds, average=None, zero_division=0)
        recall_per_class = recall_score(y_eval, preds, average=None, zero_division=0)
        detail_lines = [
            "Confusion matrix (hang = thuc te, cot = du doan):",
            "                 du_doan_thap  du_doan_cao",
            f"thuc_te_thap     {cm[0][0]:>12} {cm[0][1]:>12}",
            f"thuc_te_cao      {cm[1][0]:>12} {cm[1][1]:>12}",
            "",
            f"Lop thu_nhap_thap (0): precision={precision_per_class[0]:.4f}, recall={recall_per_class[0]:.4f}",
            f"Lop thu_nhap_cao  (1): precision={precision_per_class[1]:.4f}, recall={recall_per_class[1]:.4f}",
        ]
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w") as f:
            f.write("\n".join(detail_lines) + "\n")

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_threshold_f1)
        mlflow.log_metric("positive_ratio", positive_ratio)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "best_threshold": best_threshold,
                    "best_threshold_f1": best_threshold_f1,
                    "positive_ratio": positive_ratio,
                },
                f,
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
