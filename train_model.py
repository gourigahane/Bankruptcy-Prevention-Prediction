import argparse
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

FEATURES = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk",
]
TARGET = "class"
MODEL_FILE = "Bankruptcy_Prevention.pkl"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    X = df[FEATURES]
    y = df[TARGET]

    
    groups = df[FEATURES].astype(str).agg("_".join, axis=1)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train(data_path: str, out_path: str = MODEL_FILE) -> LogisticRegression:
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {out_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and pickle the bankruptcy prediction model.")
    parser.add_argument(
        "--data",
        default="Bankruptcy_prevention_Dataset.csv",
        help="Path to the semicolon-delimited dataset CSV.",
    )
    parser.add_argument("--out", default=MODEL_FILE, help="Output path for the pickle file.")
    args = parser.parse_args()

    train(args.data, args.out)
