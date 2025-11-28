import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from pathlib import Path

def read_params(params_path: str):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def main(params_path: str = "params.yaml"):
    params = read_params(params_path)

    processed_dir = Path(params["data"]["processed_dir"])

    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/raw/data.csv")

    df = df.dropna()
    df = df.drop_duplicates()
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train saved to: {train_path}")
    print(f"Test saved to: {test_path}")


if __name__ == "__main__":
    main()