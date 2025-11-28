import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def read_params(params_path: str):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def main(params_path: str = "params.yaml"):
    params = read_params(params_path)

    # загрузка данных
    processed_dir = Path(params["data"]["processed_dir"])
    target_col = params["data"]["target_col"]

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]


    # создание модели
    C = params["model"]["C"]
    max_iter = params["model"]["max_iter"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter))
        ]
    )

    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]

    # логирование через MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model_type", "logistic_regression")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        print(f"Accuracy: {acc:.4f}")

        model_path = Path("model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(model, artifact_path="model")

        mlflow.log_artifact(str(model_path))


if __name__ == "__main__":
    main()